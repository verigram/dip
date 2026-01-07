import io
from datetime import datetime
from typing import List, Optional, Dict

import pytest
from httpx import Request
from pytest_httpx import HTTPXMock
from urllib.parse import urljoin

from src.predict import RemoteModel, InvalidResponseStatusException, BatchPredictStat, BatchRetryStat, DataPointRetryStat


def filter_requests(requests: List[Request], url: Optional[str]=None, method: Optional[str]=None,
                    content: Optional[bytes]=None, headers: Dict[bytes, bytes]={}) -> List[Request]:
    ans = []

    for req in requests:
        if url is not None and url != req.url:
            continue

        if method is not None and method != req.method:
            continue

        if content is not None and content != req.content:
            continue

        req_headers = {k:v for k,v in req.headers.raw}
        headers_check_passed = True
        for header_name in headers:
            if header_name not in req_headers or req_headers[header_name] != headers[header_name]:
                headers_check_passed = False
                break
        if not headers_check_passed:
            continue

        ans.append(req)

    return ans


@pytest.fixture
def base_url():
    return "http://localhost"


@pytest.fixture
def model(base_url):
    return RemoteModel(url=base_url + "/liveness", skip_revision_for_single_point=False)


@pytest.mark.asyncio
async def test__input_must_be_list(model):
    with pytest.raises(ValueError):
        await model.batch_predict(x="not_list")


@pytest.fixture
def responses_1(base_url, model, httpx_mock: HTTPXMock):
    revision_url = urljoin(base_url, '/model_revision')
    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})

    httpx_mock.add_response(method="POST", status_code=200, url=model.url, match_headers={"x-model-revision": "1"},
                            match_content=b"data1", content=b"result")


@pytest.mark.asyncio
async def test__remote_model__batch_predict__simple_case(model, responses_1):

    res, stat = await model.batch_predict(x=["data1"])

    assert len(res) == 1


# ----------------------------
@pytest.fixture
def responses_3(base_url, model, httpx_mock: HTTPXMock):
    revision_url = urljoin(base_url, '/model_revision')
    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})

    httpx_mock.add_response(method="POST", status_code=200, url=model.url, match_headers={"x-model-revision": "1"},
                            match_content=b"data1", content=b"result1_1")
    httpx_mock.add_response(method="POST", status_code=200, url=model.url, match_headers={"x-model-revision": "1"},
                            match_content=b"data2", content=b"result1_2")
    httpx_mock.add_response(method="POST", status_code=200, url=model.url, match_headers={"x-model-revision": "1"},
                            match_content=b"data3", content=b"result1_3")


@pytest.mark.asyncio
async def test__remote_model__batch_predict__batch_of_3_all_success(model, responses_3):

    res, stat = await model.batch_predict(x=["data1", "data2", "data3"])

    assert len(res) == 3
    assert res[0] == b"result1_1"
    assert res[1] == b"result1_2"
    assert res[2] == b"result1_3"


# ----------------------------
@pytest.fixture
def responses_with_some_failures_1(base_url, model, httpx_mock: HTTPXMock):
    revision_url = urljoin(base_url, '/model_revision')
    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})

    httpx_mock.add_response(method="POST", status_code=400, url=model.url, match_headers={"x-model-revision": "1"},
                            match_content=b"data1")

    httpx_mock.add_response(method="POST", status_code=200, url=model.url, match_headers={"x-model-revision": "1"},
                            match_content=b"data1", content=b"result1_1")
    httpx_mock.add_response(method="POST", status_code=200, url=model.url, match_headers={"x-model-revision": "1"},
                            match_content=b"data2", content=b"result1_2")


@pytest.mark.asyncio
async def test__remote_model__batch_predict__batch_of_2__1_failure__no_revision_update(model,
                                                                                       responses_with_some_failures_1):

    res, stat = await model.batch_predict(x=["data1", "data2"])

    assert len(res) == 2
    assert res[0] == b"result1_1"
    assert res[1] == b"result1_2"


# ----------------------------
@pytest.fixture
def model_retry_1(base_url):
    return RemoteModel(url=base_url + "/predict", data_point_retry=1,
                       skip_revision_for_single_point=False)


@pytest.fixture
def model_retry_2(base_url):
    return RemoteModel(url=base_url + "/predict", data_point_retry=2,
                       skip_revision_for_single_point=False)


@pytest.mark.asyncio
async def test__remote_model__batch_predict__batch_of_2___1_failure__1_revision_update(model_retry_1, httpx_mock,
                                                                                       base_url):
    revision_url = urljoin(base_url, '/model_revision')

    # batch predict 1
    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_1.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1")
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_1.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data2", content=b"result1_2")

    # batch predict 2
    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "2"})
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_1.url,
                            match_headers={"x-model-revision": "2"}, match_content=b"data1", content=b"result2_1")
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_1.url,
                            match_headers={"x-model-revision": "2"}, match_content=b"data2", content=b"result2_2")

    res, stat = await model_retry_1.batch_predict(x=["data1", "data2"])

    assert len(res) == 2
    assert res[0] == b"result2_1"
    assert res[1] == b"result2_2"

    requests = httpx_mock.get_requests()

    assert len(requests) == 6

    assert len(filter_requests(requests[0:1], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[1:3], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'1'})) == 1
    assert len(filter_requests(requests[1:3], url='http://localhost/predict', method='POST', content=b'data2',
                               headers={b'x-model-revision': b'1'})) == 1

    assert len(filter_requests(requests[3:4], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[4:], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'2'})) == 1
    assert len(filter_requests(requests[4:], url='http://localhost/predict', method='POST', content=b'data2',
                               headers={b'x-model-revision': b'2'})) == 1


@pytest.mark.asyncio
async def test__remote_model__batch_predict__batch_of_1__2_failures__2_revision_updates(model_retry_1, httpx_mock,
                                                                                        base_url):
    revision_url = urljoin(base_url, '/model_revision')

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_1.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1")

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "2"})
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_1.url,
                            match_headers={"x-model-revision": "2"}, match_content=b"data1")

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "3"})
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_1.url,
                            match_headers={"x-model-revision": "3"}, match_content=b"data1", content=b"result3_1")

    res, stat = await model_retry_1.batch_predict(x=["data1"])

    assert len(res) == 1
    assert res[0] == b"result3_1"

    requests = httpx_mock.get_requests()

    assert len(requests) == 6

    assert len(filter_requests(requests[0:1], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[1:2], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'1'})) == 1

    assert len(filter_requests(requests[2:3], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[3:4], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'2'})) == 1

    assert len(filter_requests(requests[4:5], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[5:6], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'3'})) == 1


@pytest.mark.asyncio
async def test__remote_model__batch_predict__batch_of_1__2_failures__2_revision_updates_but_same_revision(model_retry_1,
                                                                                                 httpx_mock, base_url):
    revision_url = urljoin(base_url, '/model_revision')

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_1.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1")

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_1.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1")

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_1.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1", content=b"result1_1")

    res, stat = await model_retry_1.batch_predict(x=["data1"])

    assert len(res) == 1
    assert res[0] == b"result1_1"

    requests = httpx_mock.get_requests()

    assert len(requests) == 6

    assert len(filter_requests(requests[0:1], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[1:2], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'1'})) == 1

    assert len(filter_requests(requests[2:3], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[3:4], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'1'})) == 1

    assert len(filter_requests(requests[4:5], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[5:6], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'1'})) == 1


@pytest.mark.asyncio
async def test__remote_model__batch_predict__batch_of_1__2_failures__2_revision_updates_revision_comeback(model_retry_1,
                                                                                                 httpx_mock, base_url):
    revision_url = urljoin(base_url, '/model_revision')

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_1.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1")

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "2"})
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_1.url,
                            match_headers={"x-model-revision": "2"}, match_content=b"data1")

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_1.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1", content=b"result1_1")

    res, stat = await model_retry_1.batch_predict(x=["data1"])

    assert len(res) == 1
    assert res[0] == b"result1_1"

    requests = httpx_mock.get_requests()

    assert len(requests) == 6

    assert len(filter_requests(requests[0:1], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[1:2], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'1'})) == 1

    assert len(filter_requests(requests[2:3], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[3:4], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'2'})) == 1

    assert len(filter_requests(requests[4:5], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[5:6], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'1'})) == 1


@pytest.mark.asyncio
async def test__remote_model__batch_predict__any_httpx_client_exception__should_retry(model_retry_2, httpx_mock,
                                                                                       base_url):
    revision_url = urljoin(base_url, '/model_revision')

    # batch predict 1
    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})
    httpx_mock.add_exception(exception=Exception('ho'), method="POST", url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1")
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1", content=b"result1_1")

    res, stat = await model_retry_2.batch_predict(x=["data1"])

    assert len(res) == 1
    assert res[0] == b"result1_1"

    requests = httpx_mock.get_requests()
    assert len(requests) == 3


@pytest.mark.asyncio
async def test__remote_model__batch_predict__batch_of_1__5_failures__2_revision_updates(model_retry_2, httpx_mock,
                                                                                        base_url):
    revision_url = urljoin(base_url, '/model_revision')

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1")
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1")

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "2"})
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "2"}, match_content=b"data1")
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "2"}, match_content=b"data1")

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "3"})
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "3"}, match_content=b"data1", content=b'wrong_result')
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "3"}, match_content=b"data1", content=b"result3_1")

    res, stat = await model_retry_2.batch_predict(x=["data1"])

    assert len(res) == 1
    assert res[0] == b"result3_1"

    requests = httpx_mock.get_requests()

    assert len(requests) == 9

    assert len(filter_requests(requests[0:1], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[1:2], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'1'})) == 1
    assert len(filter_requests(requests[2:3], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'1'})) == 1

    assert len(filter_requests(requests[3:4], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[4:5], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'2'})) == 1
    assert len(filter_requests(requests[5:6], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'2'})) == 1

    assert len(filter_requests(requests[6:7], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[7:8], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'3'})) == 1
    assert len(filter_requests(requests[8:9], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'3'})) == 1


@pytest.mark.asyncio
async def test__remote_model__batch_predict__batch_of_2__7_failures__2_revision_updates(model_retry_2, httpx_mock,
                                                                                        base_url):
    revision_url = urljoin(base_url, '/model_revision')

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1")
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data2")
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1")
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data2")

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "2"})
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "2"}, match_content=b"data1", content=b'wrong_result')
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "2"}, match_content=b"data2")
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "2"}, match_content=b"data2")

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "3"})
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "3"}, match_content=b"data1", content=b"wrong_result")
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "3"}, match_content=b"data1", content=b"result3_1")
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "3"}, match_content=b"data2", content=b"result3_2")

    res, stat = await model_retry_2.batch_predict(x=["data1", "data2"])

    assert len(res) == 2
    assert res[0] == b"result3_1"
    assert res[1] == b"result3_2"

    requests = httpx_mock.get_requests()

    assert len(requests) == 13

    assert len(filter_requests(requests[0:1], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[1:5], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'1'})) == 2
    assert len(filter_requests(requests[1:5], url='http://localhost/predict', method='POST', content=b'data2',
                               headers={b'x-model-revision': b'1'})) == 2
    #
    assert len(filter_requests(requests[5:6], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[6:9], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'2'})) == 1
    assert len(filter_requests(requests[6:9], url='http://localhost/predict', method='POST', content=b'data2',
                               headers={b'x-model-revision': b'2'})) == 2
    #
    assert len(filter_requests(requests[9:10], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[10:], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'3'})) == 2
    assert len(filter_requests(requests[10:], url='http://localhost/predict', method='POST', content=b'data2',
                               headers={b'x-model-revision': b'3'})) == 1


@pytest.mark.asyncio
async def test__remote_model__batch_predict__batch_of_2__6_failures__2_revision_updates_same_revision(model_retry_2,
                                                                                                      httpx_mock,
                                                                                                      base_url):
    revision_url = urljoin(base_url, '/model_revision')

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1")
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data2")
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1")
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data2")

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1", content=b'result2_1')
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data2")
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data2")

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})
    # httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
    #                         match_headers={"x-model-revision": "1"}, match_content=b"data1", content=b"wrong_result")
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data2", content=b"result3_2")

    res, stat = await model_retry_2.batch_predict(x=["data1", "data2"])

    assert len(res) == 2
    assert res[0] == b"result2_1"
    assert res[1] == b"result3_2"

    requests = httpx_mock.get_requests()

    assert len(requests) == 11

    assert len(filter_requests(requests[0:1], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[1:5], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'1'})) == 2
    assert len(filter_requests(requests[1:5], url='http://localhost/predict', method='POST', content=b'data2',
                               headers={b'x-model-revision': b'1'})) == 2
    #
    assert len(filter_requests(requests[5:6], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[6:9], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'1'})) == 1
    assert len(filter_requests(requests[6:9], url='http://localhost/predict', method='POST', content=b'data2',
                               headers={b'x-model-revision': b'1'})) == 2
    #
    assert len(filter_requests(requests[9:10], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[10:], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'1'})) == 0
    assert len(filter_requests(requests[10:], url='http://localhost/predict', method='POST', content=b'data2',
                               headers={b'x-model-revision': b'1'})) == 1


@pytest.mark.asyncio
async def test__remote_model__batch_predict__batch_of_2__n_failures__2_revision_updates_revision_comeback(model_retry_2,
                                                                                                      httpx_mock,
                                                                                                      base_url):
    revision_url = urljoin(base_url, '/model_revision')

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1", content=b"result1_1")
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data2")
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data2")

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "2"})
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "2"}, match_content=b"data1", content=b'result2_1')
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "2"}, match_content=b"data2")
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "2"}, match_content=b"data2")

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1", content=b"result3_1")
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data2", content=b'result3_2')

    res, stat = await model_retry_2.batch_predict(x=["data1", "data2"])

    assert len(res) == 2
    assert res[0] == b"result3_1"
    assert res[1] == b"result3_2"

    requests = httpx_mock.get_requests()

    assert len(requests) == 11

    assert len(filter_requests(requests[0:1], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[1:4], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'1'})) == 1
    assert len(filter_requests(requests[1:4], url='http://localhost/predict', method='POST', content=b'data2',
                               headers={b'x-model-revision': b'1'})) == 2

    assert len(filter_requests(requests[4:5], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[5:8], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'2'})) == 1
    assert len(filter_requests(requests[5:8], url='http://localhost/predict', method='POST', content=b'data2',
                               headers={b'x-model-revision': b'2'})) == 2

    assert len(filter_requests(requests[8:9], url='http://localhost/model_revision', method='GET')) == 1
    assert len(filter_requests(requests[9:], url='http://localhost/predict', method='POST', content=b'data1',
                               headers={b'x-model-revision': b'1'})) == 1
    assert len(filter_requests(requests[9:], url='http://localhost/predict', method='POST', content=b'data2',
                               headers={b'x-model-revision': b'1'})) == 1


@pytest.mark.asyncio
async def test__remote_model__batch_predict__all_retries_exceeded__should_return_none(model_retry_2,
                                                                                     httpx_mock, base_url):
    # arrange
    revision_url = urljoin(base_url, '/model_revision')

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"}, is_reusable=True)
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1", is_reusable=True)
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data2", is_reusable=True)

    #act
    res, stat = await model_retry_2.batch_predict(x=["data1", "data2"])

    # assert
    assert res is None

    # data_point_retries * data_points * batch_retries +
    # model_revision_request * batch_retries
    # = 2 * 2 * 3 + 1 * 3 = 13 requests
    requests = httpx_mock.get_requests()
    assert len(requests) == 15


@pytest.mark.asyncio
async def test__remote_model__batch_predict__revision_invalid_response__keyname(model_retry_2,
                                                                                httpx_mock, base_url):
    revision_url = urljoin(base_url, '/model_revision')

    httpx_mock.add_response(method="GET", url=revision_url, json={"revisio": "1"}, is_reusable=True)

    res, stat = await model_retry_2.batch_predict(x=["data1", "data2"])
    assert res is None

    assert len(stat.batch_retries) == 3

    for batch_retry_stat in stat.batch_retries:
        assert batch_retry_stat.revision is None
        assert batch_retry_stat.revision_error == 'ProtocolViolationException(\'No "revision" in JSON\')'

    requests = httpx_mock.get_requests()

    assert len(requests) == 3


@pytest.mark.asyncio
async def test__remote_model__batch_predict__revision_invalid_response__not_json(model_retry_2,
                                                                                 httpx_mock, base_url):
    revision_url = urljoin(base_url, '/model_revision')

    httpx_mock.add_response(method="GET", url=revision_url, content=b'revision="1"', is_reusable=True)

    res, stat = await model_retry_2.batch_predict(x=["data1", "data2"])
    assert res is None

    assert len(stat.batch_retries) == 3

    for batch_retry_stat in stat.batch_retries:
        assert batch_retry_stat.revision is None
        assert batch_retry_stat.revision_error == "ProtocolViolationException('Response is not JSON at revision')"

    requests = httpx_mock.get_requests()

    assert len(requests) == 3


@pytest.mark.asyncio
async def test__remote_model__batch_predict__revision__return_not_200(model_retry_2,
                                                                                    httpx_mock, base_url):
    revision_url = urljoin(base_url, '/model_revision')

    statuses = [100, 199, 201, 299, 300, 400, 401, 500, 501]

    for status in statuses:
        httpx_mock.add_response(method="GET", url=revision_url, status_code=status, json={"revision": "1"})
        httpx_mock.add_response(method="GET", url=revision_url, status_code=status, json={"revision": "1"})
        httpx_mock.add_response(method="GET", url=revision_url, status_code=status, json={"revision": "1"})

    for status in statuses:
        res, stat = await model_retry_2.batch_predict(x=["data1", "data2"])

        assert res is None
        assert len(stat.batch_retries) == 3

        for batch_retry_stat in stat.batch_retries:
            assert batch_retry_stat.revision is None
            assert batch_retry_stat.revision_error == f"InvalidResponseStatusException('Received status code {status} at revision')"

    requests = httpx_mock.get_requests()

    assert len(requests) == len(statuses) * 3


@pytest.mark.asyncio
async def test__remote_model__batch_predict__revision_invalid_response__should_batch_retry(model_retry_2,
                                                                                           httpx_mock, base_url):
    revision_url = urljoin(base_url, '/model_revision')

    httpx_mock.add_response(method="GET", url=revision_url, json={"revisio": "1"})
    httpx_mock.add_response(method="GET", url=revision_url, status_code=500, json={"revision": "1"})
    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1", content=b"result1_1")
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data2", content=b"result1_2")

    res, stat = await model_retry_2.batch_predict(x=["data1", "data2"])

    assert res[0] == b"result1_1"
    assert res[1] == b"result1_2"

    requests = httpx_mock.get_requests()

    assert len(requests) == 5


@pytest.mark.skip(reason="behavior have changed")
@pytest.mark.asyncio
async def test__remote_model__batch_predict__predict__return_not_200__raise_error(model_retry_2,
                                                                                   httpx_mock, base_url):
    revision_url = urljoin(base_url, '/model_revision')

    statuses = [100, 199, 201, 299, 300, 401, 500, 501]

    for status in statuses:
        httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})
        httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                                match_headers={"x-model-revision": "1"}, match_content=b"data1", content=b"result1")
        httpx_mock.add_response(method="POST", status_code=status, url=model_retry_2.url,
                                match_headers={"x-model-revision": "1"}, match_content=b"data2", content=b"result2")

    for status in statuses:
        with pytest.raises(InvalidResponseStatusException) as exc_info:
            await model_retry_2.batch_predict(x=["data1", "data2"])
        assert str(exc_info.value) == f'Received status code {status} at predict'

    requests = httpx_mock.get_requests()

    assert len(requests) == 3 * len(statuses)


@pytest.mark.asyncio
async def test__remote_model__batch_predict__dictionary_of_files__should_be_sent_as_files(model_retry_2, httpx_mock,
                                                                                          base_url):
    revision_url = urljoin(base_url, '/model_revision')

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"}, is_reusable=True)

    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1", is_reusable=True)

    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, is_reusable=True)

    await model_retry_2.batch_predict(x=["data1", {'file2': b'content_2', 'key2': 'value2'}, {'file3': io.BytesIO(b'content_3')}])

    requests = httpx_mock.get_requests()

    assert requests[1].content == b'data1'

    assert b'data1' not in requests[2].content
    assert b'file3' not in requests[2].content
    assert b'value3' not in requests[2].content
    assert b'Content-Disposition: form-data; name="file2"; filename="upload"\r\nContent-Type: application/octet-stream\r\n\r\ncontent_2\r\n' in requests[2].content
    assert b'Content-Disposition: form-data; name="key2"\r\n\r\nvalue2\r\n' in requests[2].content

    assert b'data1' not in requests[3].content
    assert b'data1' not in requests[3].content
    assert b'Content-Disposition: form-data; name="file3"; filename="upload"\r\nContent-Type: application/octet-stream\r\n\r\ncontent_3\r\n' in requests[3].content
    assert b'file2' not in requests[3].content
    assert b'value2' not in requests[3].content


@pytest.mark.asyncio
async def test__remote_model__batch_predict__file__should_be_sent_as_file(model_retry_2, httpx_mock,
                                                                                   base_url):
    revision_url = urljoin(base_url, '/model_revision')

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"}, is_reusable=True)

    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1", is_reusable=True)

    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"}, is_reusable=True)

    await model_retry_2.batch_predict(x=["data1", b'content_2', io.BytesIO(b'content_3')])

    requests = httpx_mock.get_requests()

    assert requests[1].content == b'data1'

    assert b'data1' not in requests[2].content
    assert b'content_3' not in requests[2].content
    assert b'Content-Disposition: form-data; name="media"; filename="upload"\r\nContent-Type: application/octet-stream\r\n\r\ncontent_2\r\n' in requests[2].content

    assert b'data1' not in requests[3].content
    assert b'content_2' not in requests[3].content
    assert b'Content-Disposition: form-data; name="media"; filename="upload"\r\nContent-Type: application/octet-stream\r\n\r\ncontent_3\r\n' in requests[3].content


@pytest.mark.asyncio
async def test__remote_model__batch_predict__stats_simple_case(model_retry_2,
                                                               httpx_mock,
                                                               base_url):
    revision_url = urljoin(base_url, '/model_revision')

    # batch predict 1
    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})
    httpx_mock.add_exception(exception=Exception('ho'),
                             method="POST", url=model_retry_2.url,
                             match_headers={"x-model-revision": "1"},
                             match_content=b"data1")
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"},
                            match_content=b"data1")
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "1"},
                            match_content=b"data2", content=b"result1_2")

    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "2"})
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "2"},
                            match_content=b"data1", content=b"result2_1")
    httpx_mock.add_response(method="POST", status_code=400, url=model_retry_2.url,
                            match_headers={"x-model-revision": "2"},
                            match_content=b"data2")
    httpx_mock.add_response(method="POST", status_code=200, url=model_retry_2.url,
                            match_headers={"x-model-revision": "2"},
                            match_content=b"data2", content=b"result2_2")

    res, stat = await model_retry_2.batch_predict(x=["data1", "data2"])

    assert isinstance(stat, BatchPredictStat)
    assert len(stat.batch_retries) == 2

    batch_1 = stat.batch_retries[0]
    batch_2 = stat.batch_retries[1]

    assert isinstance(batch_1, BatchRetryStat)
    assert isinstance(batch_2, BatchRetryStat)
    assert batch_1.revision == '1'
    assert batch_2.revision == '2'
    assert batch_1.revision_error is None
    assert batch_2.revision_error is None
    assert isinstance(batch_1.start_time, datetime)
    assert isinstance(batch_1.end_time, datetime)
    assert isinstance(batch_2.start_time, datetime)
    assert isinstance(batch_2.end_time, datetime)
    assert batch_1.start_time < batch_1.end_time
    assert batch_2.start_time < batch_2.end_time
    assert batch_1.end_time < batch_2.start_time

    assert len(batch_1.data_point_retries) == 3
    assert len(batch_2.data_point_retries) == 3

    for data_point_retry in batch_1.data_point_retries + batch_2.data_point_retries:
        assert isinstance(data_point_retry, DataPointRetryStat)
        assert isinstance(data_point_retry.start_time, datetime)
        assert isinstance(data_point_retry.end_time, datetime)
        assert data_point_retry.start_time < data_point_retry.end_time
        assert data_point_retry.data_idx in [0, 1]

    def filter(retries: List[DataPointRetryStat], data_idx: int) -> List[DataPointRetryStat]:
        ans = []
        for retry in retries:
            if retry.data_idx == data_idx:
                ans.append(retry)
        return ans

    batch1_0 = filter(batch_1.data_point_retries, 0)
    assert len(batch1_0) == 2
    assert batch1_0[0].status_code is None
    assert batch1_0[0].error == "Exception('ho')"
    assert batch1_0[1].status_code == 400
    assert batch1_0[1].error is None

    batch1_1 = filter(batch_1.data_point_retries, 1)
    assert len(batch1_1) == 1
    assert batch1_1[0].status_code == 200
    assert batch1_1[0].error == None

    batch2_0 = filter(batch_2.data_point_retries, 0)
    assert len(batch2_0) == 1
    assert batch2_0[0].status_code == 200
    assert batch2_0[0].error is None

    batch2_1 = filter(batch_2.data_point_retries, 1)
    assert len(batch2_1) == 2
    assert batch2_1[0].status_code == 400
    assert batch2_1[0].error is None
    assert batch2_1[1].status_code == 200
    assert batch2_1[1].error is None


# ----------------------------
# Single data point mode tests - no model revision request
@pytest.fixture
def model_skip_revision(base_url):
    return RemoteModel(url=base_url + "/predict", skip_revision_for_single_point=True)


@pytest.mark.asyncio
async def test__remote_model__batch_predict__single_point_mode__no_revision_request(
        model_skip_revision, httpx_mock, base_url):
    """
    Test that in single point mode, the model sends data immediately without
    requesting revision
    """

    # arrange ======================
    # Only mock the POST endpoint, NOT the model_revision endpoint
    httpx_mock.add_response(method="POST", status_code=200, url=model_skip_revision.url,
                            match_headers={"x-dip": "off"}, match_content=b"data1", content=b"result1")

    # act ==========================
    res, stat = await model_skip_revision.batch_predict(x=["data1"])

    # assert =======================
    assert len(res) == 1
    assert res[0] == b"result1"

    requests = httpx_mock.get_requests()

    # Should only have 1 POST request, no GET request to /model_revision
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].url == model_skip_revision.url

    # Verify headers
    assert 'x-model-revision' not in requests[0].headers
    assert requests[0].headers['x-dip'] == 'off'


@pytest.mark.asyncio
async def test__remote_model__batch_predict__single_point_mode__with_retry(
        model_skip_revision, httpx_mock, base_url):
    """Test that single point mode still retries on failure"""
    # arrange ======================
    # First attempt fails, second succeeds
    httpx_mock.add_response(method="POST", status_code=400, url=model_skip_revision.url,
                            match_headers={"x-dip": "off"}, match_content=b"data1")
    httpx_mock.add_response(method="POST", status_code=200, url=model_skip_revision.url,
                            match_headers={"x-dip": "off"}, match_content=b"data1", content=b"result1")

    # act =========================
    res, stat = await model_skip_revision.batch_predict(x=["data1"])

    # assert ======================
    assert len(res) == 1
    assert res[0] == b"result1"

    requests = httpx_mock.get_requests()

    # Should have 2 POST requests (retry), but still no GET to /model_revision
    assert len(requests) == 2
    assert all(req.method == "POST" for req in requests)
    assert all(req.url == model_skip_revision.url for req in requests)
    assert all(req.headers['x-dip'] == 'off' for req in requests)


@pytest.mark.asyncio
async def test__remote_model__batch_predict__single_point_mode__multiple_points__uses_revision(
        model_skip_revision, httpx_mock, base_url):
    """Test that skip_revision_for_single_point=True with multiple data points still requests model revision"""

    # arrange ======================
    revision_url = urljoin(base_url, '/model_revision')

    # Mock the model revision endpoint - should be called for multiple data points
    httpx_mock.add_response(method="GET", url=revision_url, json={"revision": "1"})

    # Mock POST endpoints with x-model-revision header (not x-dip: off)
    httpx_mock.add_response(method="POST", status_code=200, url=model_skip_revision.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data1", content=b"result1")
    httpx_mock.add_response(method="POST", status_code=200, url=model_skip_revision.url,
                            match_headers={"x-model-revision": "1"}, match_content=b"data2", content=b"result2")

    # act =========================
    res, stat = await model_skip_revision.batch_predict(x=["data1", "data2"])

    # assert ======================
    assert len(res) == 2
    assert res[0] == b"result1"
    assert res[1] == b"result2"

    requests = httpx_mock.get_requests()

    # Should have 1 GET to /model_revision and 2 POST requests with x-model-revision header
    assert len(requests) == 3
    assert requests[0].method == "GET"
    assert requests[0].url == revision_url
    assert requests[1].method == "POST"
    assert requests[2].method == "POST"

    # Verify x-model-revision header is used, not x-dip
    assert requests[1].headers['x-model-revision'] == '1'
    assert requests[2].headers['x-model-revision'] == '1'
    assert 'x-dip' not in requests[1].headers
    assert 'x-dip' not in requests[2].headers

