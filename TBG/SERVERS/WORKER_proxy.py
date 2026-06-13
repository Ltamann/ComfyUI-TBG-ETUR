import sys, os, socket, struct
import threading
import time
import cloudpickle
import traceback

"""
ROOT_DIR = os.path.dirname(   # .../ComfyUI-TBG-ETUR
    os.path.dirname(          # .../TBG
        os.path.dirname(      # .../TBG/SERVERS
            os.path.abspath(__file__)
        )
    )
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
"""

# 1) Prefer explicit root from main process (works in Nuitka/frozen builds)
ROOTDIR = os.environ.get("TBGETUR_ROOTDIR")
print("WORKER ROOTDIR =", ROOTDIR, file=sys.stderr)
print(os.environ.get("TBGETUR_ROOTDIR"))
# 2) Fallback to old __file__-based logic if env var is missing
if not ROOTDIR:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    tbg_dir = os.path.dirname(this_dir)      # .../TBG
    ROOTDIR = os.path.dirname(tbg_dir)       # .../ComfyUI-TBG-ETUR

if ROOTDIR and ROOTDIR not in sys.path:
    sys.path.insert(0, ROOTDIR)

COMFY_ROOT = os.environ.get("COMFYUI_ROOT")
if COMFY_ROOT and COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)


import TBG.TBG_APP as TBG_APP
from TBG.TBG_APP.constants import (
    set_current_tiler_id,
    get_tbg,
    attach_shared_arrays_to_tbg,
    attach_plain_meta_to_tbg,
)
from TBG.SERVERS.COMFYUI_proxy import MainController

# SAFEGUARD: Ensure getport() is available on MainController
# This handles cases where the module was cached before getport() was added
if not hasattr(MainController, "getport"):
    # Dynamically add getport() if missing
    @classmethod
    def _getport_compat(cls) -> int:
        port = os.environ.get("TBG_MAIN_PORT")
        if not port:
            raise RuntimeError("TBG_MAIN_PORT not set in worker environment")
        return int(port)
    MainController.getport = _getport_compat
    print("[TBG_WORKER] Added getport() compatibility method to MainController", file=sys.stderr)

# Print immediately so we know the script started
#print(f"🔧 WORKER SCRIPT STARTED: PID={os.getpid()}", file=sys.stderr)

def start_mainrpc_watchdog() -> None:
    """
    Periodically check if the main RPC server (COMFYUI side) is still alive.
    Controlled by env var TBG_MAIN_WATCHDOG_INTERVAL (seconds).
    Set <=0 or unset to disable.
    """
    interval_raw = os.environ.get("TBG_MAIN_WATCHDOG_INTERVAL", "").strip()
    if not interval_raw:
        return

    try:
        interval = float(interval_raw)
    except ValueError:
        print(
            f"WORKER watchdog disabled: invalid TBG_MAIN_WATCHDOG_INTERVAL={interval_raw!r}",
            file=sys.stderr,
        )
        return

    if interval <= 0:
        return

    max_failures = 3  # how many consecutive failed checks before we shut down

    def _resolve_main_port() -> int:
        # Preferred source: environment provided by main process.
        env_port = os.environ.get("TBG_MAIN_PORT")
        if env_port:
            return int(env_port)

        # Fallback: internal accessor if available.
        get_port_fn = getattr(MainController, "_get_port", None)
        if callable(get_port_fn):
            return int(get_port_fn())

        # Last resort: backwards-compat alias.
        getport_fn = getattr(MainController, "getport", None)
        if callable(getport_fn):
            return int(getport_fn())

        raise RuntimeError("No usable main RPC port source (env/_get_port/getport)")

    def _loop():
        failures = 0
        while True:
            time.sleep(interval)
            port = None
            try:
                port = _resolve_main_port()

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1.0)
                    s.connect(("127.0.0.1", port))
                # success -> reset failure counter
                failures = 0
            except Exception as e:
                failures += 1
                print(
                    f"WORKER watchdog: main RPC check failed {failures}/{max_failures} "
                    f"(port={port if port is not None else 'unknown'}): {e}",
                    file=sys.stderr,
                )
                if failures >= max_failures:
                    print(
                        "WORKER watchdog: main RPC seems dead, shutting worker down.",
                        file=sys.stderr,
                    )
                    os._exit(0)  # hard exit to avoid hanging

    t = threading.Thread(target=_loop, daemon=True)
    t.start()



def worker_main(port: int) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", int(port)))
    server.listen(1)
    """
    print(f"✅ WORKER: Listening on port {port}", file=sys.stderr)
    """
    # NEW: start watchdog thread
    start_mainrpc_watchdog()

    while True:
        conn, addr = server.accept()
        conn.settimeout(None)  # Remove timeout completely
        """
        print(f"✅ WORKER: Connection from {addr}", file=sys.stderr)
        """
        try:
            len_bytes = conn.recv(8)
            if not len_bytes:
                conn.close()
                continue

            payload_len = struct.unpack("!Q", len_bytes)[0]
            payload_bytes = bytearray()
            while len(payload_bytes) < payload_len:
                chunk = conn.recv(
                    min(65536, payload_len - len(payload_bytes))
                )
                if not chunk:
                    break
                payload_bytes.extend(chunk)

            payload = cloudpickle.loads(payload_bytes)

            tiler_id = payload.get("tiler_id", None)
            shared_meta = payload.get("shared_meta", None)
            plain_meta = payload.get("plain_meta", None)
            set_current_tiler_id(tiler_id)

            #print(f"[TBG_WORKER] incoming shared_meta keys: {list(shared_meta.keys()) if isinstance(shared_meta, dict) else shared_meta}")

            # Rebuild this tiler's TBG from shared_meta (no GPL imports)
            if tiler_id is not None and shared_meta is not None:
                try:
                    T = get_tbg(tiler_id)
                    attach_shared_arrays_to_tbg(T, shared_meta)
                    attach_plain_meta_to_tbg(T, plain_meta)
                except Exception as e:
                    print(f"[TBG_WORKER] attach worker metadata failed for tiler {tiler_id}: {e}")

            class_name = payload["class_name"]
            method_name = payload["method_name"]
            args = payload["args"]
            kwargs = payload["kwargs"]

            try:
                import torch
                if torch.cuda.is_available():
                    print(
                        f"[TBG_WORKER][Device] job={class_name}.{method_name} tiler={tiler_id} "
                        f"cuda_count={torch.cuda.device_count()} current_cuda={torch.cuda.current_device()}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[TBG_WORKER][Device] job={class_name}.{method_name} tiler={tiler_id} device=cpu",
                        file=sys.stderr,
                    )
            except Exception as device_log_err:
                print(f"[TBG_WORKER][Device] diagnostics failed: {device_log_err}", file=sys.stderr)
            """
            print(
                f"🔧 WORKER: Calling {class_name}.{method_name} "
                f"with {len(args)} args, {len(kwargs)} kwargs",
                file=sys.stderr,
            )
            """
            try:
                worker_class = getattr(TBG_APP, class_name)
                method = getattr(worker_class, method_name)
                result = method(*args, **kwargs)
            except Exception as e:
                print(
                    f"❌ WORKER ERROR in {class_name}.{method_name}: {e}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
                result = e  # send exception back

            # Preflight: try pickling each arg individually
            for i, arg in enumerate(payload["args"]):
                try:
                    cloudpickle.dumps(arg)
                except Exception as e:
                    print(f"❌ PREFLIGHT FAIL: args[{i}] of type {type(arg)} cannot be pickled: {e}")
                    raise

            result_bytes = cloudpickle.dumps(result)
            conn.sendall(struct.pack("!Q", len(result_bytes)))
            conn.sendall(result_bytes)

        except Exception as e:
            print(f"❌ WORKER ERROR (outer): {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        finally:
            conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        """
        print("Usage: WORKER_proxy.py <port>", file=sys.stderr)
        """
        sys.exit(1)

    worker_main(sys.argv[1])
