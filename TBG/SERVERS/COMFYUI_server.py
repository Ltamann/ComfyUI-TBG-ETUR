"""
COMFYUI_server.py

Main-process RPC server for callbacks from the worker.

Usage:

    from main_rpc import register_main_class

    @register_main_class
    class SomeComfyClass:
        @staticmethod
        def do_something(x):
            return x + 1

The worker can then call:

    COMFYUI.SomeComfyClass.do_something(123)
"""

import socket
import struct
import threading
import traceback
from typing import Any, Dict, Type

import cloudpickle


MAIN_CLASSES: Dict[str, Type[Any]] = {}


def register_main_class(cls: Type[Any]) -> Type[Any]:
    """
    Register a class to be callable from the worker via RPC.

    The class will be looked up by its __name__, so on the worker side you
    use the same class name when calling COMFYUI.<ClassName>.<method>(...).
    """
    MAIN_CLASSES[cls.__name__] = cls
    #print(f"✅ Registered main RPC class: {cls.__name__}")
    return cls


def main_rpc_server(port: int) -> None:
    """
    Simple TCP server that listens for RPC calls from the worker.
    Protocol is identical to the worker-side one:
    - 8-byte big-endian length
    - cloudpickle'd dict {class_name, method_name, args, kwargs}
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", int(port)))
    server.listen(5)
    #print(f"🔌 MAIN RPC: Listening on port {port}")

    while True:
        conn, addr = server.accept()
        t = threading.Thread(
            target=_handle_conn,
            args=(conn, addr),
            daemon=True,
        )
        t.start()


def _handle_conn(conn: socket.socket, addr) -> None:
    conn.settimeout(None)  # Remove timeout completely
    try:
        len_bytes = conn.recv(8)
        if not len_bytes:
            return

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
        class_name = payload["class_name"]
        method_name = payload["method_name"]
        args = payload["args"]
        kwargs = payload["kwargs"]
        """
        print(
            f"🔌 MAIN RPC: {class_name}.{method_name} "
            f"with {len(args)} args, {len(kwargs)} kwargs "
            f"from {addr}"
        )
        """
        try:
            cls = MAIN_CLASSES.get(class_name)
            if cls is None:
                raise RuntimeError(f"Unknown main RPC class: {class_name}")

            method = getattr(cls, method_name)
            result = method(*args, **kwargs)
        except Exception as e:
            print(
                f"❌ MAIN RPC ERROR in {class_name}.{method_name}: {e}"
            )
            traceback.print_exc()
            result = e

        result_bytes = cloudpickle.dumps(result)
        conn.sendall(struct.pack("!Q", len(result_bytes)))
        conn.sendall(result_bytes)
    except Exception as e:
        print(f"❌ MAIN RPC ERROR (outer): {e}")
        traceback.print_exc()
    finally:
        try:
            conn.close()
        except Exception:
            pass
