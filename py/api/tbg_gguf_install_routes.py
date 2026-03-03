import asyncio
import threading

from aiohttp import web
from server import PromptServer

from ..services.gguf_installer import install_optional_gguf_runtime


_INSTALL_LOCK = threading.Lock()
_INSTALL_RUNNING = False


@PromptServer.instance.routes.post("/TBG/install_gguf_runtime")
async def tbg_install_gguf_runtime(_request):
    global _INSTALL_RUNNING

    with _INSTALL_LOCK:
        if _INSTALL_RUNNING:
            return web.json_response(
                {
                    "ok": False,
                    "status": "busy",
                    "message": "GGUF runtime installation is already running.",
                    "requires_restart": False,
                    "manual_command": None,
                }
            )
        _INSTALL_RUNNING = True

    try:
        result = await asyncio.to_thread(install_optional_gguf_runtime)
        return web.json_response(result)
    finally:
        with _INSTALL_LOCK:
            _INSTALL_RUNNING = False
