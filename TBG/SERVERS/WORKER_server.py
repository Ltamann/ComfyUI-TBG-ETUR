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

        max_retries = 4  # total attempts = max_retries + 1
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            # Always start from a clean slate
            cls.cleanup_worker()
            cls._worker_process = None
            cls._worker_port = None

            # Ensure main RPC is ready before we spawn worker (so we can pass port)
            cls._ensure_main_rpc()
            if cls._main_rpc_port is None:
                raise RuntimeError("Main RPC port not initialized")

            # Pick free port for worker
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                cls._worker_port = s.getsockname()[1]

            # Start worker subprocess
            env = os.environ.copy()

            # Compute plugin root: .../ComfyUI-TBG-ETUR
            this_dir = os.path.dirname(os.path.abspath(__file__))
            tbg_dir = os.path.dirname(this_dir)  # .../TBG
            plugin_root = os.path.dirname(tbg_dir)  # .../ComfyUI-TBG-ETUR


            # Start worker subprocess
            env = os.environ.copy()
            env["TBGETUR_ROOTDIR"] = plugin_root
            env["TBG_WORKER_PORT"] = str(cls._worker_port)
            env["TBG_MAIN_PORT"] = str(cls._main_rpc_port)

            # ComfyUI root: .../ComfyUI (two levels above custom_nodes/ComfyUI-TBG-ETUR)
            comfy_root = os.path.dirname(os.path.dirname(plugin_root))  # /workspace/ComfyUI
            env["COMFYUI_ROOT"] = comfy_root

            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            bridge_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "WORKER_proxy.py",
            )

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
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )

                t = threading.Thread(
                    target=_pipe_worker_output,
                    args=(cls._worker_process,),
                    daemon=True,
                )
                t.start()

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
        """Terminate the worker subprocess (if any) and wait for exit."""
        if cls._worker_process:
            try:
                cls._worker_process.terminate()
            except Exception:
                pass
            try:
                cls._worker_process.wait(timeout=5)
            except Exception:
                pass

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

        data = cloudpickle.dumps(payload)

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(None)  # Remove timeout completely
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
                chunk = client.recv(
                    min(65536, result_len - len(result_bytes))
                )
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
