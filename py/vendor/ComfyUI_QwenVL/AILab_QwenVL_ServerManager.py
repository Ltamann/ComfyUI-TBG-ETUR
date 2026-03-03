import os
import shlex
import subprocess
import sys
import time
import shutil
import socket
from urllib import parse


class QwenVLServerManager:
    def __init__(self):
        self.process = None
        self.command_signature = None

    @staticmethod
    def _normalize_base_url(base_url):
        base = (base_url or "http://127.0.0.1:11434/v1").strip().rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    @staticmethod
    def _split_cmd(command_line):
        return shlex.split(command_line, posix=(os.name != "nt"))

    def _build_command(self, cfg, model_cfg):
        base_url = self._normalize_base_url(cfg.get("base_url"))
        parsed = parse.urlparse(base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        template = (cfg.get("command") or "python -m llama_cpp.server").strip()
        args = (cfg.get("args") or "").strip()
        lowered_template = template.lower()
        if lowered_template.startswith("python ") or lowered_template == "python":
            # Force current Comfy Python runtime to keep dependency/environment consistent.
            template = f"{sys.executable} {template[6:].strip()}".strip()
        if lowered_template.startswith("python.exe ") or lowered_template == "python.exe":
            template = f"{sys.executable} {template[10:].strip()}".strip()
        if lowered_template == "llama-server" and shutil.which("llama-server") is None:
            template = f"{sys.executable} -m llama_cpp.server"
        substitutions = {
            "model_path": model_cfg.get("model_path", ""),
            "mmproj_path": model_cfg.get("mmproj_path", ""),
            "server_model": model_cfg.get("server_model", ""),
            "host": host,
            "port": str(port),
            "base_url": base_url,
        }
        for key, value in substitutions.items():
            template = template.replace("{" + key + "}", str(value))
            args = args.replace("{" + key + "}", str(value))

        cmdline = f"{template} {args}".strip()
        lower = cmdline.lower()
        if model_cfg.get("model_path") and " -m " not in f" {lower} " and " --model " not in f" {lower} ":
            cmdline = f'{cmdline} -m "{model_cfg["model_path"]}"'
        if model_cfg.get("mmproj_path") and "mmproj" not in lower and "clip-model-path" not in lower:
            cmdline = f'{cmdline} --mmproj "{model_cfg["mmproj_path"]}"'
        return cmdline

    def _is_running(self):
        return self.process is not None and self.process.poll() is None

    @staticmethod
    def _is_port_free(host, port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.2)
                return s.connect_ex((host, int(port))) != 0
        except Exception:
            return False

    def _find_free_port(self, host, start_port, max_tries=30):
        port = int(start_port)
        for _ in range(max_tries):
            if self._is_port_free(host, port):
                return port
            port += 1
        return None

    def stop(self, reason="manual", timeout_s=8):
        if not self._is_running():
            self.process = None
            self.command_signature = None
            return False
        print(f"[QwenVL-Server] Stopping managed server reason='{reason}'")
        self.process.terminate()
        start = time.time()
        while time.time() - start < timeout_s:
            if self.process.poll() is not None:
                self.process = None
                self.command_signature = None
                print(f"[QwenVL-Server] Managed server stopped in {time.time() - start:.2f}s")
                return True
            time.sleep(0.1)
        self.process.kill()
        self.process = None
        self.command_signature = None
        print("[QwenVL-Server] Managed server force-killed after timeout")
        return True

    def ensure_ready(self, cfg, model_cfg, client):
        launch_mode = (cfg.get("launch_mode") or "managed_local").strip()
        startup_timeout_s = float(cfg.get("startup_timeout_s", 45))
        if launch_mode == "external_only":
            status, _ = client.health_check()
            if status < 200 or status >= 300:
                raise RuntimeError(
                    f"[QwenVL-Server] Server health failed stage=health mode=external_only "
                    f"url='{cfg.get('base_url')}' status={status}"
                )
            print("[QwenVL-Server] External server health check passed")
            return

        # Safety: in managed_local mode, never hijack an already-running endpoint
        # we do not own. This avoids talking to / stopping unrelated services.
        try:
            status, _ = client.health_check()
            endpoint_alive = 200 <= status < 300
        except Exception:
            endpoint_alive = False
        if endpoint_alive and not self._is_running():
            parsed = parse.urlparse(self._normalize_base_url(cfg.get("base_url")))
            host = parsed.hostname or "127.0.0.1"
            current_port = parsed.port or (443 if parsed.scheme == "https" else 80)
            next_port = self._find_free_port(host, current_port + 1)
            if next_port is None:
                raise RuntimeError(
                    "[QwenVL-Server] Safety check failed: target endpoint is already alive and "
                    "no free fallback port found. Use launch_mode='external_only' or change base URL."
                )
            new_base = f"{parsed.scheme}://{host}:{next_port}/v1"
            cfg["base_url"] = new_base
            client.reconfigure_base_url(new_base, cfg.get("health_endpoint", "/v1/models"))
            print(
                f"[QwenVL-Server] Target endpoint already in use by non-ETUR process. "
                f"Auto-fallback to managed port {next_port}."
            )

        cmdline = self._build_command(cfg, model_cfg)
        signature = f"{cmdline}|{client.health_url}"

        if self._is_running() and self.command_signature == signature:
            return
        if self._is_running() and self.command_signature != signature:
            self.stop("config_changed")

        argv = self._split_cmd(cmdline)
        print(f"[QwenVL-Server] Starting managed server cmd='{cmdline}'")
        self.process = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        self.command_signature = signature

        started = time.time()
        last_err = None
        while time.time() - started < startup_timeout_s:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"[QwenVL-Server] Server start failed stage=start exit_code={self.process.returncode} cmd='{cmdline}'"
                )
            try:
                status, _ = client.health_check()
                if 200 <= status < 300:
                    print(f"[QwenVL-Server] Health ready in {time.time() - started:.2f}s status={status}")
                    return
            except Exception as exc:
                last_err = exc
            time.sleep(0.5)

        raise RuntimeError(
            f"[QwenVL-Server] Server start failed stage=health timeout={startup_timeout_s}s cmd='{cmdline}' "
            f"last_error={last_err}"
        )


_SERVER_MANAGER = QwenVLServerManager()


def get_server_manager():
    return _SERVER_MANAGER
