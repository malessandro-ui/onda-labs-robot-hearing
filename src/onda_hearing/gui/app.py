"""Dependency-free localhost server for the Onda acoustic-scene workbench."""
from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import webbrowser

from .backend import GuiInputError, capabilities, simulate_scenario


STATIC_DIR = Path(__file__).with_name("static")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_REQUEST_BYTES = 1_000_000


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "OndaWorkbench/0.17"

    def _json(self, status: HTTPStatus, value: object) -> None:
        body = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?", 1)[0]
        if path == "/api/capabilities":
            self._json(HTTPStatus.OK, capabilities())
            return
        static = STATIC_FILES.get(path)
        if static is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        filename, content_type = static
        body = (STATIC_DIR / filename).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; media-src 'self' blob: data:; connect-src 'self'; img-src 'self' data:")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/api/simulate":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise GuiInputError("request body is empty or too large")
            payload = json.loads(self.rfile.read(length))
            self._json(HTTPStatus.OK, simulate_scenario(payload))
        except (GuiInputError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # fail visibly without exposing a traceback in the browser
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"simulation failed: {type(exc).__name__}: {exc}"})

    def log_message(self, format: str, *args) -> None:
        print(f"[onda-gui] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Onda acoustic-scene workbench")
    parser.add_argument("--host", default="127.0.0.1", help="bind address; defaults to localhost only")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be within 1..65535")
    server = ThreadingHTTPServer((args.host, args.port), WorkbenchHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Onda Hearing Workbench: {url}")
    print("Press Ctrl-C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
