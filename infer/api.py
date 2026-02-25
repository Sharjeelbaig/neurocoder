"""Minimal HTTP API for TinyMoE task execution."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any

from infer.schemas import SchemaValidationError, TaskRequest
from infer.service import HeuristicModelAdapter, TaskService


class TinyMoERequestHandler(BaseHTTPRequestHandler):
    service = TaskService(adapter=HeuristicModelAdapter())

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/task":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        content_length = self.headers.get("Content-Length")
        if content_length is None:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "missing Content-Length"})
            return

        try:
            body = self.rfile.read(int(content_length))
            payload = json.loads(body.decode("utf-8"))
            request = TaskRequest.from_dict(payload)
            response = self.service.handle(request)
            self._write_json(HTTPStatus.OK, response.to_dict())
        except json.JSONDecodeError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {exc}"})
        except SchemaValidationError as exc:
            self._write_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive service boundary
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"internal error: {exc}"})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        # Keep API output deterministic and quiet in tests.
        return


def run_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    server = ThreadingHTTPServer((host, port), TinyMoERequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
