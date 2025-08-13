import json
import os
import urllib.parse

import requests
from aiohttp import web
from server import PromptServer
from .py.utils.constants import get_apiurl

ROUTES_ALREADY_REGISTERED = False

def register_routes():
    global ROUTES_ALREADY_REGISTERED
    if ROUTES_ALREADY_REGISTERED:
        return
    ROUTES_ALREADY_REGISTERED = True

    routes = web.RouteTableDef()

    @routes.get("/custom/patreon/login")
    async def login_handler(request):
        return web.Response(text="Login page")

    @routes.get("/custom/patreon/callback")
    async def callback_handler(request):
        return web.Response(text="Callback page")

    PromptServer.instance.app.add_routes(routes)

class PatreonAuthServer:
    TOKEN_FILE = os.path.join(os.path.dirname(__file__), "token_cache.json")
    ALLOWED_TIERS = {"Gold", "Premium"}

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.CLIENT_ID = client_id
        self.CLIENT_SECRET = client_secret
        self.REDIRECT_URI = redirect_uri

    def register_routes(self):
        routes = web.RouteTableDef()

        @routes.get("/custom/patreon/login")
        async def login_handler(request):
            query = {
                "response_type": "code",
                "client_id": self.CLIENT_ID,
                "redirect_uri": self.REDIRECT_URI,
                "scope": "identity identity.memberships"
            }
            redirect_url = "https://www.patreon.com/oauth2/authorize?" + urllib.parse.urlencode(query)
            raise web.HTTPFound(location=redirect_url)

        @routes.get("/custom/patreon/callback")
        async def callback_handler(request):
            query = request.rel_url.query
            code = query.get("code", None)

            if not code:
                return web.Response(status=400, text="<h2>Missing code</h2>")

            try:
                token_res = requests.post(
                    "https://www.patreon.com/api/oauth2/token",
                    data={
                        "code": code,
                        "grant_type": "authorization_code",
                        "client_id": self.CLIENT_ID,
                        "client_secret": self.CLIENT_SECRET,
                        "redirect_uri": self.REDIRECT_URI,
                    },
                )
                if token_res.status_code != 200:
                    return web.Response(
                        status=500,
                        text=f"Token exchange failed: {token_res.text}",
                    )

                with open(self.TOKEN_FILE, "w") as f:
                    json.dump(token_res.json(), f)

                return web.Response(text="<h2>Patreon login successful</h2>")

            except Exception as e:
                return web.Response(status=500, text=f"<h2>Error: {e}</h2>")

        PromptServer.instance.app.add_routes(routes)



class PatreonStatusCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "check_status"

    def check_status(self, api_token):
        # from ....utils.constants import get_apiurl
        apiurl = get_apiurl(0)

        headers = {
            "User-Agent": "Mozilla/5.0 (ComfyUI Patreon Node)"
        }
       # try:
        url = f"{apiurl}status.php?token={api_token}"

        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code != 200:
            return (f"❌ Failed: {r.status_code}",)

        data = r.json()
        status = data.get("status", "unknown")
        membership = data.get("membership", "none")
        user_id = data.get("user_id", "none")
        TBG_pro = data.get("TBG_pro", "none")

        return (f"✅ Status: {status} ({membership}) {user_id} Pro:{TBG_pro}", TBG_pro)

       # except Exception as e:
       #     return (f"❌ Error: {e}",)


NODE_CLASS_MAPPINGS = {
    "PatreonStatusCheck": PatreonStatusCheck,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "PatreonStatusCheck": "Patreon Status Check",
}

