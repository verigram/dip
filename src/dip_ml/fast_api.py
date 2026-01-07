
from fastapi import Request, Response
from starlette.responses import JSONResponse


class DipMiddlewareException(Exception):
    pass


class DipMiddleware:

    __times_init_app_called = 0

    def __init__(self, paths: list[str], app_revision: str):
        self.__paths = paths
        self.__app_revision = app_revision

    def init_app(self, app):
        if self.__times_init_app_called > 0:
            raise DipMiddlewareException('You can set dip middleware '
                                         'only once in application')

        self.__times_init_app_called += 1

        @app.middleware("http")
        async def process_request(request: Request, call_next) -> Response:

            if request.url.path == '/model_revision' and request.method == "GET":
                return JSONResponse(content={"revision": self.__app_revision},
                                    status_code=200)

            if request.url.path not in self.__paths or request.method != "POST":
                return await call_next(request)

            # Check if x-dip: off header is present (bypasses revision check)
            if 'x-dip' in request.headers and request.headers['x-dip'] == 'off':
                return await call_next(request)

            if 'x-model-revision' not in request.headers:
                return JSONResponse(
                    content={"error": "header x-model-revision is required"},
                    status_code=412)

            if request.headers['x-model-revision'] != self.__app_revision:
                return Response(status_code=400)

            return await call_next(request)
