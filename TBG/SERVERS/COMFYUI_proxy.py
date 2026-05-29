"""
COMFYUI_proxy.py

Worker-side proxy to call back into the main process, using COMFYUI.Class.method(...)
syntax.

Requires the main process to:
- Start main_rpc.main_rpc_server(...) on some port.
- Pass that port to the worker via the TBG_MAIN_PORT environment variable
  (done in TBG_Controller.ensure_worker).
"""

import os
import socket
import struct
import time
from typing import Any

import cloudpickle

import torch  # new

# --- Large tensor protection like in TBG_APP.py ---

def _compress_tensor_for_rpc(x):
    if not isinstance(x, torch.Tensor):
        return x
    if x.ndim not in (3, 4):  # only image-like tensors
        return x
    if x.dtype != torch.float32:
        return x

    nbytes = x.element_size() * x.nelement()
    MAX_UNCOMPRESSED_BYTES = 5 * 1024 * 1024 * 1024  # 5 GiB

    if nbytes <= MAX_UNCOMPRESSED_BYTES:
        return x

    compressed = x.to(torch.float16)
    print(
        f"TBGMAINRPC Compressed tensor for RPC "
        f"{nbytes/1024**3:.2f}GiB float32 -> "
        f"{compressed.numel()*compressed.element_size()/1024**3:.2f}GiB float16 "
        f"shape={x.shape}"
    )
    return compressed


def _compress_structure(obj):
    """Recursively downcast large float32 image tensors inside containers."""
    if isinstance(obj, torch.Tensor):
        return _compress_tensor_for_rpc(obj)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_compress_structure(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _compress_structure(v) for k, v in obj.items()}
    return obj

class MainController:
    # Class-level cache for the port to avoid repeated env lookups
    _port_cache: int | None = None
    _connect_retry_delays = (0.25, 0.5, 1.0, 2.0, 4.0)

    @classmethod
    def _get_port(cls) -> int:
        if cls._port_cache is not None:
            return cls._port_cache
        port = os.environ.get("TBG_MAIN_PORT")
        if not port:
            raise RuntimeError("TBG_MAIN_PORT not set in worker environment")
        cls._port_cache = int(port)
        return cls._port_cache

    # 👇 ADD THIS for compatibility with WORKER_proxy watchdog
    @classmethod
    def getport(cls) -> int:
        """
        Backwards‑compat alias used by WORKER_proxy watchdog.
        Kept separate so existing code using _get_port() still works.
        """
        return cls._get_port()

    @classmethod
    def call_main_method_old(
        cls,
        class_name: str,
        method_name: str,
        *args,
        **kwargs,
    ) -> Any:
        port = cls._get_port()

        payload = {
            "class_name": class_name,
            "method_name": method_name,
            "args": args,
            "kwargs": kwargs,
        }

        # Optional: preflight pickle like in main
        for i, arg in enumerate(payload["args"]):
            cloudpickle.dumps(arg)  # will raise if not picklable

        data = cloudpickle.dumps(payload)

        client = cls._connect_with_retry(port, class_name, method_name)
        try:
            client.sendall(struct.pack("!Q", len(data)))
            client.sendall(data)

            len_bytes = client.recv(8)
            if not len_bytes:
                raise RuntimeError("Main RPC closed connection without response")

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
                # Remote exception: re-raise locally
                raise result
            return result
        finally:
            client.close()

    @classmethod
    def _connect_with_retry(cls, port: int, class_name: str, method_name: str) -> socket.socket:
        rpc_name = f"{class_name}.{method_name}"
        last_error = None

        for attempt, delay in enumerate((0.0,) + cls._connect_retry_delays, start=1):
            if delay > 0:
                time.sleep(delay)

            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(None)
            try:
                client.connect(("127.0.0.1", int(port)))
                if attempt > 1:
                    print(f"[TBG_MAIN_RPC] Connected for {rpc_name} after retry {attempt - 1}")
                return client
            except OSError as e:
                last_error = e
                try:
                    client.close()
                except Exception:
                    pass
                if attempt <= len(cls._connect_retry_delays):
                    print(
                        f"[TBG_MAIN_RPC] Connect failed for {rpc_name} "
                        f"(attempt {attempt}/{len(cls._connect_retry_delays) + 1}): {e}; retrying"
                    )
                    continue

        print(f"[TBG_MAIN_RPC] Connect failed permanently for {rpc_name}: {last_error}")
        raise last_error

    @classmethod
    def call_main_method(
        cls,
        class_name: str,
        method_name: str,
        *args,
        **kwargs,
    ) -> Any:
        port = cls._get_port()

        payload = {
            "class_name": class_name,
            "method_name": method_name,
            "args": args,
            "kwargs": kwargs,
        }

        # Optional: preflight pickle like in main
        for i, arg in enumerate(payload["args"]):
            cloudpickle.dumps(arg)  # will raise if not picklable

        # Compress large tensors in the payload (usually small for TBGState.get, but cheap)
        safe_payload = _compress_structure(payload)

        data = cloudpickle.dumps(safe_payload)

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(None)
        client.connect(("127.0.0.1", port))
        try:
            client.sendall(struct.pack("!Q", len(data)))
            client.sendall(data)

            len_bytes = client.recv(8)
            if not len_bytes:
                raise RuntimeError("Main RPC closed connection without response")

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
                # Remote exception: re-raise locally
                raise result

            # CRITICAL: compress any huge float32 images / tensors coming back
            result = _compress_structure(result)
            return result
        finally:
            client.close()

class _MainClassProxy:
    def __init__(self, class_name: str):
        self.class_name = class_name

    def __getattr__(self, method_name: str):
        def _caller(*args, **kwargs):
            return MainController.call_main_method(
                self.class_name,
                method_name,
                *args,
                **kwargs,
            )

        return _caller


class _COMFYUI_NS:
    """
    Namespace that exposes main-process classes to the worker:

        COMFYUI.SomeComfyClass.some_method(...)
    """

    def __getattr__(self, class_name: str):
        # e.g. COMFYUI.SomeComfyClass → "SomeComfyClass"
        return _MainClassProxy(class_name)


COMFYUI = _COMFYUI_NS()
