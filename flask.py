"""
ConnectTel Telecommunications - Standalone Zero-Dependency Micro-Flask Fallback
Provides a drop-in replacement for Flask when running in environments where Flask is not installed.
If real Flask is installed in the global environment, it seamlessly delegates to it.
"""

import sys
import os
import json
import time

# Try delegating to real Flask if available in site-packages
_real_flask = None
try:
    # Temporarily remove current directory from sys.path to avoid importing self
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    orig_path = list(sys.path)
    clean_path = [p for p in orig_path if os.path.abspath(p) not in [curr_dir, os.path.abspath('.')]]
    sys.path = clean_path
    import flask as _rf
    _real_flask = _rf
finally:
    sys.path = orig_path

if _real_flask is not None and hasattr(_real_flask, "Flask"):
    Flask = _real_flask.Flask
    request = _real_flask.request
    jsonify = _real_flask.jsonify
    Response = getattr(_real_flask, "Response", None)
else:
    # Pure-Python Standalone Implementation (Zero External Dependencies)
    class LocalRequest:
        def __init__(self):
            self.method = "GET"
            self._json_data = None
            self.path = "/"
            self.headers = {}

        def get_json(self, force=True, silent=True):
            return self._json_data

    request = LocalRequest()

    def jsonify(*args, **kwargs):
        if args and kwargs:
            raise TypeError("jsonify takes either args or kwargs, not both")
        if args:
            data = args[0] if len(args) == 1 else list(args)
        else:
            data = kwargs
        return data

    class LocalResponse:
        def __init__(self, data=None, status_code=200, content_type="application/json"):
            self.status_code = status_code
            self.data_raw = data
            self.content_type = content_type
            if isinstance(data, (dict, list)):
                self.text = json.dumps(data)
                self.data = self.text.encode('utf-8')
            elif isinstance(data, str):
                self.text = data
                self.data = data.encode('utf-8')
            elif isinstance(data, bytes):
                self.data = data
                self.text = data.decode('utf-8', errors='ignore')
            else:
                self.text = ""
                self.data = b""

        def get_json(self):
            if isinstance(self.data_raw, (dict, list)):
                return self.data_raw
            try:
                return json.loads(self.text)
            except Exception:
                return None

    class LocalTestClient:
        def __init__(self, app):
            self.app = app

        def get(self, path):
            return self.app._dispatch("GET", path, None)

        def post(self, path, json=None, data=None):
            return self.app._dispatch("POST", path, json if json is not None else data)

    class Flask:
        def __init__(self, import_name=None):
            self.import_name = import_name
            self.routes = {} # (path, method) -> handler
            self.error_handlers = {}

        def route(self, rule, methods=None):
            if methods is None:
                methods = ["GET"]
            def decorator(f):
                for m in methods:
                    self.routes[(rule, m.upper())] = f
                return f
            return decorator

        def errorhandler(self, code):
            def decorator(f):
                self.error_handlers[code] = f
                return f
            return decorator

        def test_client(self):
            return LocalTestClient(self)

        def _dispatch(self, method, path, json_data):
            clean_path = path.split("?")[0]
            handler = self.routes.get((clean_path, method.upper()))
            
            if not handler:
                allowed = [m for (r, m) in self.routes.keys() if r == clean_path]
                if allowed:
                    err_handler = self.error_handlers.get(405)
                    body = err_handler(None) if err_handler else {"status": "error", "message": "Method not allowed"}
                    return LocalResponse(body, status_code=405)
                err_handler = self.error_handlers.get(404)
                body = err_handler(None) if err_handler else {"status": "error", "message": "Endpoint not found"}
                return LocalResponse(body, status_code=404)

            # Update request global
            request.method = method.upper()
            request._json_data = json_data
            request.path = path

            try:
                res = handler()
                if isinstance(res, tuple):
                    body, code = res
                else:
                    body, code = res, 200

                if isinstance(body, LocalResponse):
                    body.status_code = code
                    return body
                return LocalResponse(body, status_code=code)
            except Exception as e:
                err_handler = self.error_handlers.get(500)
                body = err_handler(e) if err_handler else {"status": "error", "message": str(e)}
                return LocalResponse(body, status_code=500)

        def __call__(self, environ, start_response):
            """Standard WSGI compliance for wsgiref, gunicorn, etc."""
            method = environ.get("REQUEST_METHOD", "GET")
            path = environ.get("PATH_INFO", "/")
            json_data = None
            if method == "POST":
                try:
                    content_len = int(environ.get("CONTENT_LENGTH", 0))
                    if content_len > 0:
                        body_raw = environ["wsgi.input"].read(content_len).decode("utf-8")
                        json_data = json.loads(body_raw)
                except Exception:
                    json_data = None

            res = self._dispatch(method, path, json_data)
            status_text = f"{res.status_code} OK" if res.status_code == 200 else f"{res.status_code} Error"
            start_response(status_text, [("Content-Type", res.content_type), ("Content-Length", str(len(res.data)))])
            return [res.data]

        def run(self, host="0.0.0.0", port=5000, debug=False):
            from wsgiref.simple_server import make_server
            print(f"Starting ConnectTel Production Server on http://{host}:{port}...")
            server = make_server(host, port, self)
            server.serve_forever()
