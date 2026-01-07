import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from src.fast_api import DipMiddleware, DipMiddlewareException

app = FastAPI()


@app.get("/")
async def root():
    return b'root'


@app.post("/")
async def root():
    return b'root'


@app.post("/liveness")
async def liveness(request: Request):
    payload = await request.body()
    revision = request.headers.get('x-model-revision', 'no-revision')

    return f'liveness_{revision}_{payload.decode("utf-8")[-1]}'


@app.get("/liveness")
async def liveness():
    return b'liveness_get'


@app.post("/face_matching")
async def liveness(request: Request):
    payload = await request.body()
    revision = request.headers['x-model-revision']

    return f'matching{revision}_{payload.decode("utf-8")[-1]}'


@app.get("/not_dip")
async def not_dip():
    return "is_not_dip"


@app.post("/not_dip")
async def not_dip():
    return "is_not_dip"


dip_paths = ["/predict", "/liveness", "/face_matching"]
dip = DipMiddleware(dip_paths, app_revision="revision_1")
dip.init_app(app)


client = TestClient(app)


def test__middleware__root_not_affected():
    response_get = client.get("/")
    response_post = client.post("/")

    assert response_get.status_code == 200
    assert 'x-model-revision' is not response_get.headers
    assert response_get.content == b'"root"'

    assert response_post.status_code == 200
    assert 'x-model-revision' is not response_post.headers
    assert response_post.content == b'"root"'


def test__middleware__not_included_paths_not_affected():
    response_get = client.get("/not_dip")
    response_post = client.post("/not_dip")
    response_post_with_header = client.post("/not_dip", headers={'x-model-revision': 'revision_1'})
    response_post_not_existant = client.post("/not_existant", headers={'x-model-revision': 'revision_1'})

    assert response_get.status_code == 200
    assert 'x-model-revision' is not response_get.headers
    assert response_get.content == b'"is_not_dip"'

    assert response_post.status_code == 200
    assert 'x-model-revision' is not response_post.headers
    assert response_post.content == b'"is_not_dip"'

    assert response_post_with_header.status_code == 200
    assert 'x-model-revision' is not response_post_with_header.headers
    assert response_post_with_header.content == b'"is_not_dip"'

    assert response_post_not_existant.status_code == 404
    assert 'x-model-revision' is not response_post_with_header.headers


def test__middleware__included_paths__get_method__not_affected():
    response_1 = client.get("/liveness")
    response_2 = client.get("/face_matching")

    assert response_1.status_code == 200
    assert 'x-model-revision' is not response_1.headers
    assert response_1.content == b'"liveness_get"'

    assert response_2.status_code == 405
    assert 'x-model-revision' is not response_2.headers


def test__middleware__revision():
    response = client.get("/model_revision")

    assert response.status_code == 200
    assert response.json() == {"revision": "revision_1"}


def test__middleware__revision_post_should_return_404():
    response = client.post("/model_revision")

    assert response.status_code == 404


def test__middleware__predict__no_revision_header_should_return_412():
    response = client.post("/liveness")

    assert response.status_code == 412
    assert response.json() == {"error": "header x-model-revision is required"}


def test__middleware__predict__matching_revision():
    response = client.post("/liveness", data="data1", headers={"x-model-revision": "revision_1"})

    assert response.status_code == 200
    assert response.content == b'"liveness_revision_1_1"'


def test__middleware__predict__non_matching_revision():
    response = client.post("/liveness", data="data1", headers={"x-model-revision": "revision_2"})

    assert response.status_code == 400


def test__middleware__using_middleware_second_time_should_raise_exception():
    with pytest.raises(DipMiddlewareException) as exc_info:
        dip.init_app(app)
    assert str(exc_info.value) == 'You can set dip middleware only once in application'


def test__middleware__predict__with_x_dip_off_header_should_succeed():
    """Test that x-dip: off header bypasses revision check on DIP paths"""
    response = client.post("/liveness", data="data1", headers={"x-dip": "off"})

    assert response.status_code == 200
    assert response.content == b'"liveness_no-revision_1"'
