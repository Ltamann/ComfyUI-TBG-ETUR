import os
import shlex
import subprocess
import sys
import time
import shutil
import socket
import tempfile
from urllib import parse


class QwenVLServerManager:
    def __init__(self):
        self.process = None
        self.command_signature = None
        self._stderr_handle = None
        self._stderr_path = None

    @staticmethod
    def _normalize_base_url(base_url):
        base = (base_url or "http://127.0.0.1:11434/v1").strip().rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    @staticmethod
    def _split_cmd(command_line):
        return shlex.split(command_line, posix=(os.name != "nt"))

    @staticmethod
    def _argv_to_cmdline(argv):
        if os.name == "nt":
            return subprocess.list2cmdline(argv)
        return shlex.join(argv)

    @staticmethod
    def _server_arg_start(argv):
        if not argv:
            return 0
        exe = os.path.basename(argv[0]).lower()
        is_python = exe.startswith("python")
        if is_python and len(argv) >= 3 and argv[1] == "-m":
            # python -m <module> : ignore interpreter -m
            return 3
        return 1

    @staticmethod
    def _has_flag(argv, start_idx, flag_names):
        names = set(flag_names)
        for i in range(start_idx, len(argv)):
            tok = argv[i]
            if tok in names:
                return True
            for name in names:
                if tok.startswith(name + "="):
                    return True
        return False

    @staticmethod
    def _stderr_tail_from_file(path, max_lines=20, max_chars=2000):
        if not path or not os.path.exists(path):
            return ""
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except Exception:
            data = b""
        if not data:
            return ""
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="ignore")
        else:
            text = str(data)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        tail = "\n".join(lines[-max_lines:])
        return tail[-max_chars:]

    def _close_stderr_capture(self):
        try:
            if self._stderr_handle is not None and not self._stderr_handle.closed:
                self._stderr_handle.close()
        except Exception:
            pass
        self._stderr_handle = None
        self._stderr_path = None

    @staticmethod
    def _is_python_llama_cpp_server(argv):
        return (
            len(argv) >= 3
            and argv[1] == "-m"
            and argv[2] == "llama_cpp.server"
        )

    def _sanitize_python_server_args(self, argv, start_idx):
        """
        Normalize args for python module server variant:
        - drop unsupported llama-server flags (--jinja, --fit, --parallel N)
        - normalize/drop flash-attn syntax for this CLI variant
        - convert legacy --mmproj to --clip_model_path
        """
        sanitized = []
        i = 0
        dropped = []

        def _consume_optional_value(idx):
            j = idx + 1
            if j < len(argv) and not argv[j].startswith("-"):
                return j + 1
            return idx + 1

        while i < len(argv):
            tok = argv[i]
            if i >= start_idx:
                if tok == "--mmproj":
                    sanitized.append("--clip_model_path")
                    i += 1
                    continue
                if tok.startswith("--mmproj="):
                    val = tok.split("=", 1)[1]
                    sanitized.append(f"--clip_model_path={val}")
                    i += 1
                    continue
                if tok == "--jinja" or tok == "--fit":
                    dropped.append(tok)
                    i = _consume_optional_value(i)
                    continue
                if tok.startswith("--fit="):
                    dropped.append("--fit")
                    i += 1
                    continue
                if tok == "--parallel":
                    dropped.append(tok)
                    i = _consume_optional_value(i)
                    continue
                if tok.startswith("--parallel="):
                    dropped.append("--parallel")
                    i += 1
                    continue
                # python -m llama_cpp.server parser in this environment does not expose --flash-attn.
                if tok in ("--flash-attn", "--flash_attn"):
                    dropped.append("--flash-attn")
                    i = _consume_optional_value(i)
                    continue
                if tok.startswith("--flash-attn=") or tok.startswith("--flash_attn="):
                    dropped.append("--flash-attn")
                    i += 1
                    continue
            sanitized.append(tok)
            i += 1
        if dropped:
            print(f"[QwenVL-Server] Dropped unsupported python server flags: {sorted(set(dropped))}")
        return sanitized

    def _build_command(self, cfg, model_cfg):
        base_url = self._normalize_base_url(cfg.get("base_url"))
        parsed = parse.urlparse(base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        template = (cfg.get("command") or "python -m llama_cpp.server").strip()
        model_args = model_cfg.get("server_args", "")
        if isinstance(model_args, list):
            model_args = " ".join(str(x) for x in model_args if str(x).strip())
        model_args = (model_args or "").strip()
        user_args = (cfg.get("args") or "").strip()
        # Merge precedence: model profile args first, then Labs/user args (user can override by appending flags).
        args = f"{model_args} {user_args}".strip()
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
        argv = self._split_cmd(cmdline)
        start = self._server_arg_start(argv)
        if self._is_python_llama_cpp_server(argv):
            argv = self._sanitize_python_server_args(argv, start)
            start = self._server_arg_start(argv)

        # Mandatory: ensure exact resolved local GGUF file is passed as model.
        model_path = model_cfg.get("model_path")
        has_model = self._has_flag(argv, start, {"--model", "-m"})
        if model_path and not has_model:
            argv.extend(["--model", str(model_path)])

        # Keep mmproj injection when missing.
        mmproj_path = model_cfg.get("mmproj_path")
        has_mmproj = self._has_flag(argv, start, {"--mmproj", "--clip-model-path", "--clip_model_path"})
        if mmproj_path and not has_mmproj:
            # python module server expects --clip_model_path
            if self._is_python_llama_cpp_server(argv):
                argv.extend(["--clip_model_path", str(mmproj_path)])
            else:
                argv.extend(["--mmproj", str(mmproj_path)])

        # Keep server binding aligned with configured base URL.
        has_host = self._has_flag(argv, start, {"--host"})
        has_port = self._has_flag(argv, start, {"--port"})
        if not has_host:
            argv.extend(["--host", str(host)])
        if not has_port:
            argv.extend(["--port", str(port)])

        return self._argv_to_cmdline(argv)

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
            self._close_stderr_capture()
            return False
        print(f"[QwenVL-Server] Stopping managed server reason='{reason}'")
        self.process.terminate()
        start = time.time()
        while time.time() - start < timeout_s:
            if self.process.poll() is not None:
                self.process = None
                self.command_signature = None
                self._close_stderr_capture()
                print(f"[QwenVL-Server] Managed server stopped in {time.time() - start:.2f}s")
                return True
            time.sleep(0.1)
        self.process.kill()
        self.process = None
        self.command_signature = None
        self._close_stderr_capture()
        print("[QwenVL-Server] Managed server force-killed after timeout")
        return True

    def ensure_ready(self, cfg, model_cfg, client):
        launch_mode = (cfg.get("launch_mode") or "managed_local").strip()
        startup_timeout_s = float(cfg.get("startup_timeout_s", 120))
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
        self._close_stderr_capture()
        stderr_tmp = tempfile.NamedTemporaryFile(prefix="tbg_qwenvl_server_", suffix=".log", delete=False)
        stderr_tmp.close()
        self._stderr_path = stderr_tmp.name
        self._stderr_handle = open(self._stderr_path, "ab")
        self.process = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_handle,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        self.command_signature = signature

        started = time.time()
        last_err = None
        while time.time() - started < startup_timeout_s:
            if self.process.poll() is not None:
                stderr_tail = self._stderr_tail_from_file(self._stderr_path)
                raise RuntimeError(
                    f"[QwenVL-Server] Server start failed stage=start exit_code={self.process.returncode} cmd='{cmdline}' "
                    f"stderr_tail='{stderr_tail}'"
                )
            try:
                status, _ = client.health_check()
                if 200 <= status < 300:
                    print(f"[QwenVL-Server] Health ready in {time.time() - started:.2f}s status={status}")
                    return
            except Exception as exc:
                last_err = exc
            time.sleep(0.5)

        stderr_tail = self._stderr_tail_from_file(self._stderr_path)
        raise RuntimeError(
            f"[QwenVL-Server] Server start failed stage=health timeout={startup_timeout_s}s cmd='{cmdline}' "
            f"last_error={last_err} stderr_tail='{stderr_tail}'"
        )


_SERVER_MANAGER = QwenVLServerManager()


def get_server_manager():
    return _SERVER_MANAGER

