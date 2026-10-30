"""`gallery serve`: the dashboard on 127.0.0.1, read-only for files, with one write route for picks, discards and notes."""
import json
import sys
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .gallery import GALLERY_DIR, gallery_index, gallery_set, load_gallery

_WRITE_LOCK = threading.Lock()


class _Handler(SimpleHTTPRequestHandler):
    gdir = None
    port = None

    def log_message(self, *args):
        pass

    def _host_ok(self):
        return self.headers.get("Host") in (f"127.0.0.1:{self.port}", f"localhost:{self.port}")

    def _json(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def do_GET(self):
        if not self._host_ok():
            return self.send_error(403)
        path = self.path.split("?", 1)[0]
        if path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", f"/{GALLERY_DIR}/index.html")
            return self.end_headers()
        if path == "/api/manifest":
            with _WRITE_LOCK:
                m = gallery_index(self.gdir)[0]
            return self._json(200, {"updated": m.get("updated"), "entries": [
                {k: v for k, v in e.items() if not k.startswith("_")} for e in m["entries"].values()]})
        from urllib.parse import unquote
        rel = unquote(path).lstrip("/")
        allowed = [GALLERY_DIR] + load_gallery(self.gdir)["sources"]
        if not any(rel == s or rel.startswith(s.rstrip("/") + "/") for s in allowed) or ".." in rel.split("/") \
                or any(part.startswith(".") and part not in (".thumbs", ".history") for part in rel.split("/")):
            return self.send_error(404)
        real = (self.gdir.parent / rel).resolve()  # a symlink must not lead out of the gallery
        if not any(real == (self.gdir.parent / s).resolve() or (self.gdir.parent / s).resolve() in real.parents
                   for s in allowed):
            return self.send_error(404)
        return super().do_GET()

    def do_POST(self):
        # a custom header forces a CORS preflight, which this server never answers: other sites can't post here
        if not self._host_ok() or self.headers.get("X-Gptkit") != "1" or self.path != "/api/state":
            return self.send_error(403)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(min(n, 65536)) or b"{}")
            with _WRITE_LOCK:
                e = gallery_set(self.gdir, self.gdir.parent / body["path"], body.get("state"), body.get("note"))
        except SystemExit as e:  # `pick` exits with its message
            return self._json(400, {"error": str(e.code)})
        except (ValueError, KeyError, TypeError, OSError) as e:
            return self._json(400, {"error": str(e)})
        entries = load_gallery(self.gdir)["entries"]
        same = [{k: v for k, v in x.items() if not k.startswith("_")} for x in entries.values()
                if x.get("unit") == e.get("unit") and x.get("group") == e.get("group")]
        return self._json(200, {"entry": {k: v for k, v in e.items() if not k.startswith("_")}, "group": same})

    def do_OPTIONS(self):
        self.send_error(405)


class _Server(ThreadingHTTPServer):
    request_queue_size = 128  # the default 5 resets a page's burst of thumbnail requests
    daemon_threads = True


def serve(gdir, port=8765, open_browser=False):
    gdir = Path(gdir).resolve()
    handler = type("GalleryHandler", (_Handler,), {"gdir": gdir, "port": port})
    httpd = _Server(("127.0.0.1", port), partial(handler, directory=str(gdir.parent)))
    url = f"http://127.0.0.1:{port}/{GALLERY_DIR}/index.html"
    print(f"gallery: {url}  (Ctrl-C stops it; picks and notes are saved to {gdir / 'manifest.json'})", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("gallery: stopped", file=sys.stderr)
    finally:
        httpd.server_close()
