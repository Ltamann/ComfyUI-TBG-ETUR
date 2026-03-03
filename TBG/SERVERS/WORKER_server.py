"""
comfyui_bridge.py / WORKER_server.py

Main-process controller that:
- Spawns a persistent worker subprocess running WORKER_proxy.py.
- Provides WORKER.Class.method(...) RPC into the worker.
- Starts a main-process RPC server (COMFYUI_server.py) for callbacks from the worker.
"""

import os
import sys
import socket
import struct
import atexit
import time
import threading
import subprocess
import signal
import tempfile
from pathlib import Path
from typing import Any

import cloudpickle  # pip install cloudpickle
from .COMFYUI_server import main_rpc_server
from ..CALLBACKS.constants import get_tbg


class TBG_Controller:
    _worker_process: subprocess.Popen | None = None
    _worker_port: int | None = None
    _worker_started_during_init: bool = False

    # track whether main RPC server has been started
    _main_rpc_started: bool = False
    _main_rpc_port: int | None = None

    # Global worker idle / shutdown state (shared by all tilers)
    _worker_shutdown_timer: threading.Timer | None = None
    _worker_last_activity: float = 0.0
    _worker_pidfile: Path | None = None

    @classmethod
    def _pidfile_dir(cls) -> Path:
        p = Path(tempfile.gettempdir()) / "TBG"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def _pidfile_path_for_port(cls, port: int) -> Path:
        return cls._pidfile_dir() / f"worker_pid_{int(port)}.txt"

    @classmethod
    def _write_pidfile(cls, pid: int, port: int, bridge_script: str) -> None:
        path = cls._pidfile_path_for_port(port)
        path.write_text(
            f"{int(pid)}\n{int(port)}\n{bridge_script}\n",
            encoding="utf-8",
        )
        cls._worker_pidfile = path

    @classmethod
    def _remove_pidfile(cls, path: Path | None = None) -> None:
        target = path or cls._worker_pidfile
        if target is None:
            return
        try:
            if target.exists():
                target.unlink()
        except Exception:
            pass
        if path is None or path == cls._worker_pidfile:
            cls._worker_pidfile = None

    @classmethod
    def _is_pid_alive(cls, pid: int) -> bool:
        try:
            if pid <= 0:
                return False
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    @classmethod
    def _get_process_cmdline(cls, pid: int) -> str:
        try:
            if sys.platform == "win32":
                ps_cmd = (
                    f"(Get-CimInstance Win32_Process -Filter \"ProcessId={int(pid)}\").CommandLine"
                )
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_cmd],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                out = (result.stdout or "").strip()
                if out:
                    return out
                return ""
            result = subprocess.run(
                ["ps", "-p", str(int(pid)), "-o", "args="],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return (result.stdout or "").strip()
        except Exception:
            return ""

    @classmethod
    def _is_tbg_worker_pid(cls, pid: int, expected_script: str | None = None) -> bool:
        cmdline = cls._get_process_cmdline(pid)
        if not cmdline:
            return False
        if "WORKER_proxy.py" not in cmdline:
            return False
        if expected_script and expected_script not in cmdline:
            return False
        return True

    @classmethod
    def _kill_verified_pid(cls, pid: int, expected_script: str | None = None) -> bool:
        if pid <= 0:
            return False
        if not cls._is_pid_alive(pid):
            return True
        if not cls._is_tbg_worker_pid(pid, expected_script=expected_script):
            print(
                f"[TBG_MAIN] Skip killing PID {pid}: process identity not verified as TBG worker."
            )
            return False
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(int(pid))],
                    capture_output=True,
                    timeout=5,
                )
            else:
                try:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGTERM)
                    deadline = time.time() + 3
                    while time.time() < deadline:
                        if not cls._is_pid_alive(pid):
                            return True
                        time.sleep(0.1)
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    os.kill(pid, signal.SIGKILL)
            return not cls._is_pid_alive(pid)
        except Exception as e:
            print(f"[TBG_MAIN] Failed to kill stale worker PID {pid}: {e}")
            return False

    @classmethod
    def _reap_stale_worker_pidfiles(cls) -> None:
        for path in cls._pidfile_dir().glob("worker_pid_*.txt"):
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                if not lines:
                    cls._remove_pidfile(path)
                    continue
                pid = int(lines[0].strip())
                expected_script = lines[2].strip() if len(lines) > 2 else None

                if not cls._is_pid_alive(pid):
                    cls._remove_pidfile(path)
                    continue

                if cls._kill_verified_pid(pid, expected_script=expected_script):
                    cls._remove_pidfile(path)
            except Exception:
                # Corrupt pidfile or parse failure: remove and continue.
                cls._remove_pidfile(path)

    @staticmethod
    def _is_retryable_worker_error(exc: Exception) -> bool:
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return True
        if isinstance(exc, OSError):
            winerror = getattr(exc, "winerror", None)
            if winerror in {10053, 10054, 10061}:
                return True
            if exc.errno in {32, 54, 104, 107, 111}:
                return True
        return False

    @classmethod
    def _rpc_roundtrip(cls, worker_port: int, payload: dict) -> Any:
        data = cloudpickle.dumps(payload)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(None)
        client.connect(("127.0.0.1", int(worker_port)))
        try:
            client.sendall(struct.pack("!Q", len(data)))
            client.sendall(data)
            len_bytes = client.recv(8)
            if not len_bytes:
                raise RuntimeError("Worker closed connection without response")
            result_len = struct.unpack("!Q", len_bytes)[0]
            result_bytes = bytearray()
            while len(result_bytes) < result_len:
                chunk = client.recv(min(65536, result_len - len(result_bytes)))
                if not chunk:
                    break
                result_bytes.extend(chunk)
            result = cloudpickle.loads(result_bytes)
            if isinstance(result, Exception):
                raise result
            return result
        finally:
            client.close()

    @classmethod
    def start_worker_on_init(cls) -> None:
        """
        Start worker once during module import; blocks until ready.
        Call this from your ComfyUI __init__ if you want eager startup.
        """
        if cls._worker_started_during_init:
            return  # Already done

        cls.ensure_worker()
        cls._worker_started_during_init = True
    @classmethod
    def mark_job_started(cls) -> None:
        """
        Mark that a worker job has just started (from any tiler/refiner).

        - Cancels any pending shutdown timer.
        - Updates the global last-activity timestamp.
        """
        timer = cls._worker_shutdown_timer
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass
            cls._worker_shutdown_timer = None

        cls._worker_last_activity = time.time()

    @classmethod
    def schedule_worker_shutdown(cls, delay: float) -> None:
        """
        Schedule a worker shutdown after 'delay' seconds of idle time.

        The actual shutdown only happens if no newer job has updated
        _worker_last_activity after this schedule call.
        """
        if delay is None or delay < 0:
            delay = 0.0

        scheduled_at = time.time()

        def _maybe_close():
            # If another job ran after this schedule call, skip shutdown.
            if cls._worker_last_activity > scheduled_at:
                return
            cls.worker_close()
            cls._worker_shutdown_timer = None

        timer = threading.Timer(delay, _maybe_close)
        cls._worker_shutdown_timer = timer
        timer.start()

    @classmethod
    def _ensure_main_rpc(cls) -> None:
        """
        Start the main-process RPC server (for worker → main callbacks)
        on a free localhost port, once.
        """
        if cls._main_rpc_started and cls._main_rpc_port is not None:
            return

        # Pick a free port for main RPC
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            cls._main_rpc_port = s.getsockname()[1]

        # Start main RPC server thread
        t = threading.Thread(
            target=main_rpc_server,
            args=(cls._main_rpc_port,),
            daemon=True,
        )
        t.start()

        cls._main_rpc_started = True
        #print(f"✅ Main RPC server ready on port {cls._main_rpc_port}")

    @classmethod
    def ensure_worker(cls) -> None:
        """
        Ensure the worker subprocess is running and listening.
        Reuses an existing live worker, otherwise spawns a new one.

        Enhanced:
        - Automatic restart with limited retries if the worker
          fails to start or bind its port.
        - Longer startup timeout so slow boots do not immediately error.
        """
        # If worker exists and is alive, reuse it
        if cls._worker_process is not None and cls._worker_process.poll() is None:
            return

        # Clean stale pidfiles/workers from previous sessions before spawn attempts.
        cls._reap_stale_worker_pidfiles()

        max_retries = 4  # total attempts = max_retries + 1
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            # Always start from a clean slate
            cls.cleanup_worker()

            # Ensure main RPC is ready before we spawn worker (so we can pass port)
            cls._ensure_main_rpc()
            if cls._main_rpc_port is None:
                raise RuntimeError("Main RPC port not initialized")

            # Pick free port for worker
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                cls._worker_port = s.getsockname()[1]

            # Compute plugin root: .../ComfyUI-TBG-ETUR
            this_dir = os.path.dirname(os.path.abspath(__file__))
            tbg_dir = os.path.dirname(this_dir)  # .../TBG
            plugin_root = os.path.dirname(tbg_dir)  # .../ComfyUI-TBG-ETUR

            # Start worker subprocess
            env = os.environ.copy()
            env["TBGETUR_ROOTDIR"] = plugin_root
            env["TBG_WORKER_PORT"] = str(cls._worker_port)
            env["TBG_MAIN_PORT"] = str(cls._main_rpc_port)
            env["TBGETUR_WORKER"] = "1"

            # ComfyUI root: .../ComfyUI (two levels above custom_nodes/ComfyUI-TBG-ETUR)
            comfy_root = os.path.dirname(os.path.dirname(plugin_root))  # /workspace/ComfyUI
            env["COMFYUI_ROOT"] = comfy_root

            startupinfo = None
            creationflags = 0
            preexec_fn = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                preexec_fn = os.setsid

            bridge_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "WORKER_proxy.py",
            )
            env["TBGETUR_WORKER_SCRIPT"] = bridge_script

            def _pipe_worker_output(proc: subprocess.Popen) -> None:
                if proc.stdout is None:
                    return
                for line in iter(proc.stdout.readline, ""):
                    if not line:
                        break
                    print(f"[TBG_APP] {line.rstrip()}")

            try:
                cls._worker_process = subprocess.Popen(
                    [sys.executable, "-u", bridge_script, str(cls._worker_port)],
                    env=env,
                    startupinfo=startupinfo,
                    creationflags=creationflags,
                    preexec_fn=preexec_fn,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                cls._write_pidfile(
                    pid=cls._worker_process.pid,
                    port=int(cls._worker_port),
                    bridge_script=bridge_script,
                )

                t = threading.Thread(
                    target=_pipe_worker_output,
                    args=(cls._worker_process,),
                    daemon=True,
                )
                t.start()
                try:
                    import torch
                    if torch.cuda.is_available():
                        print(
                            f"[TBG_MAIN][Device] Worker spawned pid={cls._worker_process.pid} "
                            f"cuda_count={torch.cuda.device_count()} current_cuda={torch.cuda.current_device()}",
                        )
                    else:
                        print(f"[TBG_MAIN][Device] Worker spawned pid={cls._worker_process.pid} device=cpu")
                except Exception as device_log_err:
                    print(f"[TBG_MAIN][Device] Worker device diagnostics failed: {device_log_err}")

                # Wait until worker is listening (max 30 sec)
                start = time.time()
                timeout = 60.0  # was 10.0

                while time.time() - start < timeout:
                    # If process exited, treat as startup crash
                    if cls._worker_process.poll() is not None:
                        raise RuntimeError("TBG APP Worker not found during startup")

                    test_sock = None
                    try:
                        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        test_sock.settimeout(0.5)
                        test_sock.connect(("127.0.0.1", cls._worker_port))
                        test_sock.close()
                        # Worker is ready
                        return
                    except OSError:
                        # Not ready yet, keep waiting
                        time.sleep(0.2)
                    finally:
                        if test_sock is not None:
                            try:
                                test_sock.close()
                            except Exception:
                                pass

                # Timeout reached without successful connect
                raise RuntimeError(
                    f"TBG APP Worker did not start listening on port {cls._worker_port}, just try to run again"
                )

            except Exception as e:
                last_error = e
                cls.cleanup_worker()
                print(
                    f"[TBG_MAIN] Worker startup failed on attempt "
                    f"{attempt + 1}/{max_retries + 1}: {e}"
                )
                # If there are retries left, wait a bit and retry
                if attempt < max_retries:
                    time.sleep(1.0)
                    continue
                # No retries left: re-raise
                raise

        # Safety fallback (should not reach here)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Worker failed to start for unknown reasons")

    @classmethod
    def cleanup_worker(cls) -> None:
        """Terminate worker subprocess deterministically and clear controller state."""
        proc = cls._worker_process
        pid = proc.pid if proc is not None else None

        try:
            if proc is not None and proc.poll() is None:
                if sys.platform == "win32":
                    try:
                        # Send CTRL_BREAK to process group leader when available.
                        os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
                    except Exception:
                        pass
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # Last resort on Windows: terminate process tree.
                        try:
                            subprocess.run(
                                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                                capture_output=True,
                                timeout=5,
                            )
                        except Exception:
                            pass
                        try:
                            proc.wait(timeout=2)
                        except Exception:
                            pass
                else:
                    # Unix: terminate process group first, then force kill group.
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGTERM)
                    except Exception:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            pgid = os.getpgid(proc.pid)
                            os.killpg(pgid, signal.SIGKILL)
                        except Exception:
                            try:
                                proc.kill()
                            except Exception:
                                pass
                        try:
                            proc.wait(timeout=2)
                        except Exception:
                            pass
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass
            # Remove known pidfile and clear state even if process handle is broken.
            cls._remove_pidfile()
            cls._worker_process = None
            cls._worker_port = None
            # Best-effort cleanup for stale worker handle from earlier session.
            if pid is not None and cls._is_pid_alive(pid):
                cls._kill_verified_pid(pid)

    @classmethod
    def call_worker_method_old(
        cls,
        tiler_id,
        class_name: str,
        method_name: str,
        *args,
        **kwargs,
    ) -> Any:
        """
        Generic method to call any worker class method.

        Example:
            WORKER.MyWorkerClass.some_method(1, 2, foo="bar")
        """
        cls.ensure_worker()

        worker_port = cls._worker_port or os.environ.get("TBG_WORKER_PORT")
        if not worker_port:
            raise RuntimeError(
                "Worker port not initialized (no _worker_port and no TBG_WORKER_PORT)"
            )
        # NEW: rebuild shared-memory snapshot for this tiler_id
        shared_meta = None
        if tiler_id is not None:
            try:
                TBG = get_tbg(tiler_id)
                """
                print("[TBG_MAIN] call_worker_method TBG:",
                      id(TBG),
                      "tiler_id:", tiler_id,
                      "INPUTS.image type:", type(getattr(TBG.INPUTS, "image", None)),
                      "shape:", getattr(getattr(TBG.INPUTS, "image", None), "shape", None))
                """
                TBG.build_shared_meta()  # create SharedMemory segments
                # convert SharedArrayRef -> plain dict for socket
                shared_meta = {
                    path: {
                        "name": ref.name,
                        "shape": tuple(ref.shape),
                        "dtype": ref.dtype,
                    }
                    for path, ref in TBG._shared_meta.items()
                }
                #print(f"[TBG_MAIN] shared_meta keys for tiler {tiler_id}: {list(shared_meta.keys())}")
            except Exception as e:
                print(f"[TBG_MAIN] build_shared_meta failed for tiler {tiler_id}: {e}")
                shared_meta = {}

        payload = {
            "tiler_id": tiler_id,
            "class_name": class_name,
            "method_name": method_name,
            "args": args,   # positional args
            "kwargs": kwargs,  # keyword args
            "shared_meta": shared_meta,  # <<< IMPORTANT
        }

        # Preflight: try pickling each arg individually
        for i, arg in enumerate(payload["args"]):
            try:
                cloudpickle.dumps(arg)
            except Exception as e:
                print(
                    f"❌ PREFLIGHT FAIL: args[{i}] of type {type(arg)} "
                    f"cannot be pickled: {e}"
                )
                raise

        for rpc_attempt in range(2):
            try:
                return cls._rpc_roundtrip(int(worker_port), payload)
            except Exception as e:
                if rpc_attempt == 0 and cls._is_retryable_worker_error(e):
                    print(f"[TBG_MAIN] Worker RPC failed ({e}); restarting worker and retrying once.")
                    cls.cleanup_worker()
                    cls.ensure_worker()
                    worker_port = cls._worker_port or os.environ.get("TBG_WORKER_PORT")
                    if not worker_port:
                        raise RuntimeError("Worker restart failed: no worker port available")
                    continue
                raise

    @classmethod
    def _build_full_shared_meta(cls, tiler_id) -> dict:
        """Build FULL shared_meta (original behavior)."""
        shared_meta = None
        if tiler_id is not None:
            try:
                TBG = get_tbg(tiler_id)
                """
                print("[TBG_MAIN] call_worker_method TBG:", id(TBG), "tiler_id:", tiler_id,
                      "INPUTS.image type:", type(getattr(TBG.INPUTS, "image", None)),
                      "shape:", getattr(getattr(TBG.INPUTS, "image", None), "shape", None))
                """
                TBG.build_shared_meta()
                shared_meta = {
                    path: {"name": ref.name, "shape": tuple(ref.shape), "dtype": ref.dtype}
                    for path, ref in TBG._shared_meta.items()
                }
                #print(f"[TBG_MAIN] FULL shared_meta keys for tiler {tiler_id}: {list(shared_meta.keys())}")
            except Exception as e:
                print(f"[TBG_MAIN] build_shared_meta failed for tiler {tiler_id}: {e}")
                shared_meta = {}
        return shared_meta

    @classmethod
    def _build_selective_shared_meta(cls, tiler_id, paths: list[str]) -> dict:
        """Build shared_meta for ONLY specified paths."""
        if tiler_id is None:
            return {}

        try:
            TBG = get_tbg(tiler_id)
            TBG.build_shared_meta()  # Build full, then filter

            selective = {
                path: meta
                for path, meta in TBG._shared_meta.items()
                if any(path.startswith(p) for p in paths)
            }
            print(f"[TBG_MAIN] SELECTIVE shared_meta for tiler {tiler_id}: {list(selective.keys())}")
            return selective
        except Exception as e:
            print(f"[TBG_MAIN] selective_shared_meta failed for tiler {tiler_id}: {e}")
            return {}

    @classmethod
    def call_worker_method(
            cls,
            tiler_id,
            class_name: str,
            method_name: str,
            *args,
            **kwargs,
    ) -> Any:
        """
        Generic method to call any worker class method.

        _tbg_send_images modes (optional kwarg):
            True (default):  Send ALL images (current behavior)
            False:           Send NO images
            ["path1", "path2"]: Send ONLY specified paths (e.g. ["INPUTS.image"])
        """
        cls.ensure_worker()

        worker_port = cls._worker_port or os.environ.get("TBG_WORKER_PORT")
        if not worker_port:
            raise RuntimeError(
                "Worker port not initialized (no _worker_port and no TBG_WORKER_PORT)"
            )

        # === ENHANCED: Selective shared_meta ===
        send_images = kwargs.pop("_tbg_send_images", True)

        if isinstance(send_images, list):
            # Selective: ["INPUTS.image", "OUTPUTS.grid_images_all.0"]
            #print(f"[TBG_MAIN] Selective shared_meta for tiler {tiler_id}: {send_images}")
            shared_meta = cls._build_selective_shared_meta(tiler_id, send_images)
        elif not send_images:
            # Skip all
            #print(f"[TBG_MAIN] Skipping shared_meta for tiler {tiler_id} (send_images=False)")
            shared_meta = {}
        else:
            # Full (original behavior)
            shared_meta = cls._build_full_shared_meta(tiler_id)

        # === ORIGINAL payload + RPC (unchanged) ===
        payload = {
            "tiler_id": tiler_id,
            "class_name": class_name,
            "method_name": method_name,
            "args": args,
            "kwargs": kwargs,  # flag popped!
            "shared_meta": shared_meta,
        }

        # === YOUR ORIGINAL preflight + socket (unchanged) ===
        for i, arg in enumerate(payload["args"]):
            try:
                cloudpickle.dumps(arg)
            except Exception as e:
                print(f"❌ PREFLIGHT FAIL: args[{i}] of type {type(arg)} cannot be pickled: {e}")
                raise

        for rpc_attempt in range(2):
            try:
                return cls._rpc_roundtrip(int(worker_port), payload)
            except Exception as e:
                if rpc_attempt == 0 and cls._is_retryable_worker_error(e):
                    print(f"[TBG_MAIN] Worker RPC failed ({e}); restarting worker and retrying once.")
                    cls.cleanup_worker()
                    cls.ensure_worker()
                    worker_port = cls._worker_port or os.environ.get("TBG_WORKER_PORT")
                    if not worker_port:
                        raise RuntimeError("Worker restart failed: no worker port available")
                    continue
                raise

    @classmethod
    def worker_close(cls) -> None:
        """
        HARD RESET: free all worker memory after every call (if you choose to
        call this). Terminates the worker; next call will spawn a fresh one.
        """
        cls.cleanup_worker()
        cls._worker_process = None
        cls._worker_port = None


class _WorkerClassProxy:
    """
    Proxy for a specific worker-side class.
    """

    def __init__(self, class_name: str,tiler_id):
        self.class_name = class_name
        self.tiler_id = tiler_id
    def __getattr__(self, method_name: str):
        def _caller(*args, **kwargs):
            return TBG_Controller.call_worker_method(
                self.tiler_id,
                self.class_name,
                method_name,
                *args,
                **kwargs,
            )

        return _caller

class WORKER_NS:
    def __getattr__(self, classname: str):
        return _WorkerClassProxy(classname)

    def id(self, tiler_id):
        return WORKERNSWithId(tiler_id)

class WORKERNSWithId:
    def __init__(self, tiler_id):
        self.tiler_id = tiler_id

    def __getattr__(self, classname: str):
        return _WorkerClassProxy(classname, tiler_id=self.tiler_id)

class WORKER_NSold:
    """
    Namespace that exposes worker classes as attributes:

        WORKER.TBG_Image.some_method(...)
    """

    def __getattr__(self, class_name: str):
        # e.g. WORKER.TBG_Image → class_name = "TBG_Image"
        return _WorkerClassProxy(class_name)


WORKER = WORKER_NS()

# Clean up worker on interpreter exit
atexit.register(TBG_Controller.cleanup_worker)


