"""Verification driver for docs/DESIGN_PHONE_AUDIO.md section 8 steps 2-3.

NOT the MCP server. Loads the engines, starts PhoneAudioServer on 0.0.0.0:8772
exactly as speak_server.main() does, then serves a tiny HTTP control surface on
127.0.0.1:8773 so the verification can be driven with curl:

  GET /status                 -> same dict as the status tool
  GET /speak?text=...         -> _speak_sync(text)
  GET /listen?max=..&sil=..&start=..  -> _listen_sync(...)
  GET /converse?text=..&...   -> _converse_sync(...)

Each call runs the same sync function the MCP tools run (under the same
cross-process audio lock) and returns JSON with the result or the exception,
plus wall-clock timings. Logs go to stderr with timestamps.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("verify")

import speak_server  # noqa: E402
from speak_phone_audio import PhoneAudioServer  # noqa: E402

CTL_PORT = 8773


def _status() -> dict:
    busy, current_tool, waiting = speak_server.audio_lock.snapshot()
    return {"busy": busy, "current_tool": current_tool, "waiting": waiting,
            "audio": "phone" if speak_server.phone.connected() else "mac"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _reply(self, code: int, body: dict) -> None:
        data = json.dumps(body, indent=1).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        t0 = time.time()
        log.info("ctl %s %s", u.path, q)
        try:
            if u.path == "/status":
                out = _status()
            elif u.path == "/speak":
                seconds, route = speak_server._speak_sync(q["text"])
                out = {"result": f"spoke for {seconds:.1f}s", "audio": route}
            elif u.path == "/listen":
                text, route = speak_server._listen_sync(float(q.get("max", 120)), float(q.get("sil", 2.0)), float(q.get("start", 45)))
                out = {"result": text, "audio": route}
            elif u.path == "/converse":
                text, route = speak_server._converse_sync(q["text"], float(q.get("max", 120)), float(q.get("sil", 2.0)), float(q.get("start", 45)))
                out = {"result": text, "audio": route}
            else:
                self._reply(404, {"error": "unknown path"})
                return
            out["elapsed_s"] = round(time.time() - t0, 3)
            log.info("ctl %s done in %.3fs: %s", u.path, out["elapsed_s"], out)
            self._reply(200, out)
        except BaseException as e:  # report everything to the curl caller
            el = round(time.time() - t0, 3)
            log.info("ctl %s RAISED after %.3fs: %s: %s", u.path, el, type(e).__name__, e)
            self._reply(500, {"error": f"{type(e).__name__}: {e}", "elapsed_s": el})


def main() -> None:
    threading.Thread(target=speak_server.engines.load, name="engine-load", daemon=True).start()
    phone = PhoneAudioServer(host="0.0.0.0", port=int(os.environ.get("SPEAK_PHONE_PORT", speak_server.PHONE_PORT_DEFAULT)))
    phone.start()
    speak_server.phone = phone
    log.info("phone server up on %s:%d; control on 127.0.0.1:%d", phone.host, phone.port, CTL_PORT)
    speak_server.engines.wait()
    log.info("engines loaded")
    ThreadingHTTPServer(("127.0.0.1", CTL_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
