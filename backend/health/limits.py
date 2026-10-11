from starlette.responses import JSONResponse


class RequestTooLarge(Exception):
    pass


class BodyLimitMiddleware:
    """Also bounds streamed/chunked uploads, before multipart parsing can spool them."""

    def __init__(self, app, maximum=10 * 1024 * 1024 + 65536):
        self.app, self.maximum = app, maximum

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.maximum:
                    raise RequestTooLarge()
            return message

        try:
            await self.app(scope, bounded_receive, send)
        except RequestTooLarge:
            response = JSONResponse(
                {
                    "error": {
                        "code": "upload_too_large",
                        "message": "Each report must be 10 MB or smaller.",
                    }
                },
                status_code=413,
            )
            await response(scope, receive, send)
