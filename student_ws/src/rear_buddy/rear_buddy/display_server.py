"""The rear display: a tiny web server on the laptop that a phone (or any screen with a browser) opens full-screen.

    server = DisplayServer(port=8080)   # starts in the background
    server.show({'scene': 'car', 'text': 'Hello!', 'animation': 'breathing'})

GET /         the display page (display.html)
GET /events   Server-Sent Events: the current message at once, then every new one (the page reconnects by itself)
GET /state    the current message as JSON: scene, text, animation, qr (true = show the QR code)
GET /qr.svg   the QR code (404 without one)

Standard library only, plus segno for the QR code (optional: without it there is no QR code). The phone must be on
the same network as the laptop (home Wi-Fi, or the car's BITT Wi-Fi).
"""
import io
import json
import pathlib
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = pathlib.Path(__file__).with_name('display.html')
HEARTBEAT_S = 15.0  # keeps idle connections open through Wi-Fi routers and phone power saving


def qr_svg(text):
    """text as a QR code SVG (black on white with the standard quiet zone), or None when segno is missing."""
    try:
        import segno
    except ImportError:
        return None
    out = io.BytesIO()
    # omitsize: only a viewBox, so the page can scale it to any size. Error level M survives a bit of glare.
    segno.make(text, error='m').save(out, kind='svg', border=4, dark='#000', light='#fff', omitsize=True)
    return out.getvalue()


class DisplayServer:
    def __init__(self, host='0.0.0.0', port=8080, qr_url=None, qr_scenes=('fancy_car',)):
        self.qr = qr_svg(qr_url) if qr_url else None
        self.qr_scenes = set(qr_scenes) if self.qr else set()
        self.state = {'scene': 'nobody', 'text': '', 'animation': 'scanner', 'qr': False}
        self.version = 0
        self.changed = threading.Condition()
        self.page = PAGE.read_bytes()
        self.httpd = ThreadingHTTPServer((host, port), self._handler())
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]  # the real port when started with port=0
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def show(self, reaction):
        """reaction: dict with scene, text, animation (e.g. Reaction.__dict__). Thread-safe."""
        with self.changed:
            self.state = {k: reaction.get(k, '') for k in ('scene', 'text', 'animation')}
            self.state['qr'] = self.state['scene'] in self.qr_scenes
            self.version += 1
            self.changed.notify_all()

    def urls(self):
        return [f'http://{ip}:{self.port}/' for ip in local_ips()]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet: no line per request
                pass

            def do_GET(self):
                path = self.path.split('?')[0]
                if path == '/':
                    self._send(200, 'text/html; charset=utf-8', server.page)
                elif path == '/state':
                    with server.changed:
                        body = json.dumps(server.state).encode()
                    self._send(200, 'application/json', body)
                elif path == '/events':
                    self._events()
                elif path == '/qr.svg' and server.qr:
                    self._send(200, 'image/svg+xml', server.qr)
                else:
                    self._send(404, 'text/plain', b'not found')

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def _events(self):
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with server.changed:
                            if server.version == seen:
                                server.changed.wait(HEARTBEAT_S)
                            if server.version == seen:
                                chunk = b': heartbeat\n\n'
                            else:
                                seen = server.version
                                chunk = f'data: {json.dumps(server.state)}\n\n'.encode()
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass  # the phone went away; it reconnects by itself

        return Handler


def local_ips():
    """This machine's addresses that other devices on the network can reach (best guess, no traffic is sent)."""
    ips = set()
    for probe in ('10.42.0.1', '192.168.0.1', '8.8.8.8'):  # car Wi-Fi, typical home router, internet route
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((probe, 9))  # UDP connect only picks a route
                ips.add(s.getsockname()[0])
        except OSError:
            pass
    ips.discard('0.0.0.0')
    return sorted(ips) or ['127.0.0.1']
