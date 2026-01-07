import asyncio
import io
from datetime import UTC, datetime
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import BaseModel, Field


class ProtocolViolationException(Exception):
    pass


class RetryCountExceededException(Exception):
    pass


class InvalidResponseStatusException(Exception):
    pass


class DipResponse(BaseModel):
    data_idx: int
    status_code: int | None = None
    error: str | None = None
    content: bytes | None = None


def utcnow():
    return datetime.now(UTC)


class DataPointRetryStat(BaseModel):
    start_time: datetime = Field(default_factory=utcnow)
    end_time: datetime | None = None
    data_idx: int
    status_code: int | None = None
    error: str | None = None


class BatchRetryStat(BaseModel):
    start_time: datetime = Field(default_factory=utcnow)
    end_time: datetime | None = None
    revision: str | None = None
    revision_error: str | None = None
    data_point_retries: list[DataPointRetryStat] = []


class BatchPredictStat(BaseModel):
    batch_retries: list[BatchRetryStat] = []


class RemoteModel:
    def __init__(self, url: str,
                 data_point_retry: int = 10,
                 batch_retry: int = 3,
                 sequential: bool = False,
                 skip_revision_for_single_point: bool = True,
                 timeout: int = 60):
        """Represents a remote machine learning model accessible via a URL.

        This class handles sending data points to a remote endpoint for inference,
        managing retry logic for both individual points and full batches.
        :param url: The endpoint URL for prediction requests.
        :param data_point_retry:The number of times to retry inference for a single
        data point upon failure
        :param batch_retry: The number of times to retry the entire batch if specific
        data points fail after `data_point_retry` attempts.
        :param sequential: If True, processes data points sequentially.
        Useful for testing on single-node installations.
        :param skip_revision_for_single_point: If True, skips the model revision request
        for single data point batches and sends data immediately with
        'x-dip: off' header. For batches with multiple data points, normal DIP protocol
        is always used.
        :param timeout: Timeout for HTTP requests
        """
        self._url = url
        self._base_url = self.__parse_base_url()
        self._revision_path = "/model_revision"
        self.__batch_retry = batch_retry
        self.__data_point_retry = data_point_retry
        self.__sequential = sequential
        self.__skip_revision_for_single_point = skip_revision_for_single_point
        self.__timeout = timeout

    def __parse_base_url(self) -> str:
        parsed = urlparse(self._url)
        return f"{parsed.scheme}://{parsed.netloc}"

    @property
    def url(self) -> str:
        return self._url

    async def batch_predict(
            self, x: list) -> tuple[list[bytes | None] | None, BatchPredictStat]:
        """Batch prediction
        :param x: A list of input data points to make distributed predictions on
        :return: List of inference results
        """
        self.__check_input(x)
        batch_predict_stat = BatchPredictStat()
        predictions = [None] * len(x)
        revision = None

        for _batch_retry_idx in range(self.__batch_retry):
            batch_retry_stat = BatchRetryStat()
            batch_predict_stat.batch_retries.append(batch_retry_stat)
            prev_revision = revision

            # Only skip revision if skip_revision_for_single_point is True and
            # batch has single data point
            should_skip_revision = self.__skip_revision_for_single_point and len(x) == 1

            if not should_skip_revision:
                try:
                    revision = await self.__request_revision()
                    batch_retry_stat.revision = revision
                except Exception as e:
                    batch_retry_stat.end_time = datetime.now(UTC)
                    batch_retry_stat.revision_error = repr(e)
                    continue

                if revision != prev_revision:
                    predictions = [None] * len(x)
            else:
                # Skip revision request for a single data point - use None as revision
                revision = None
                batch_retry_stat.revision = None

            for __ in range(self.__data_point_retry):
                tasks = self.__generate_tasks(predictions, revision)

                if self.__sequential:
                    responses = await self.__predict_batch_sequentially(x, tasks)
                    dip_responses = [els[0] for els in responses]
                    data_point_retry_stats = [els[1] for els in responses]
                else:
                    gathered_responses = await self.__gather_responses(x, tasks)
                    dip_responses = [els[0] for els in gathered_responses]
                    data_point_retry_stats = [els[1] for els in gathered_responses]

                self.__assign_predictions_from_dip_responses(predictions, dip_responses)

                batch_retry_stat.data_point_retries += data_point_retry_stats

                if None not in predictions:
                    batch_retry_stat.end_time = datetime.now(UTC)
                    return predictions, batch_predict_stat

            batch_retry_stat.end_time = datetime.now(UTC)

        if None in predictions:
            return None, batch_predict_stat

        return None, batch_predict_stat

    @staticmethod
    def __check_input(x) -> None:
        if not isinstance(x, list):
            raise ValueError("Input must be a list")

    async def __request_revision(self):
        revision_path = urljoin(self._base_url, self._revision_path)
        async with httpx.AsyncClient(timeout=self.__timeout) as client:
            r = await client.get(revision_path)

        if r.status_code != 200:
            raise InvalidResponseStatusException(f'Received status code '
                                                 f'{r.status_code} at revision')

        try:
            r = r.json()
        except Exception as e:
            raise ProtocolViolationException('Response is not JSON at revision') from e

        if 'revision' not in r:
            raise ProtocolViolationException('No "revision" in JSON')

        revision = r["revision"]

        return revision

    @staticmethod
    def __generate_tasks(predictions: list, revision: str | None) -> list:
        tasks = []

        for data_idx, pred in enumerate(predictions):
            if pred is None:
                tasks.append((data_idx, revision))

        return tasks

    async def __gather_responses(
            self, x, tasks: list) -> list[tuple[DipResponse, DataPointRetryStat]]:
        return await asyncio.gather(*[self.__predict(data_idx, x[data_idx], revision)
                                      for data_idx, revision in tasks])

    async def __predict_batch_sequentially(
            self, x, tasks) -> list[tuple[DipResponse, DataPointRetryStat]]:

        responses = []
        for task in tasks:
            idx = task[0]
            revision = task[1]
            response = await self.__predict(
                data_idx=idx,
                data=x[idx],
                model_revision=revision)

            responses.append(response)

        return responses

    async def __predict(
            self, data_idx: int, data,
            model_revision: str | None) -> tuple[DipResponse, DataPointRetryStat]:
        data, files = self.__extract_files(data)

        data_point_retry_stat = DataPointRetryStat(data_idx=data_idx)

        # Build headers
        headers = {}
        if model_revision is None:
            # No revision means we're in single point mode
            # (skip_revision=True with a single data point)
            headers["x-dip"] = "off"
        else:
            headers["x-model-revision"] = model_revision

        try:
            async with httpx.AsyncClient(timeout=self.__timeout) as client:
                raw_response = await client.post(
                    self._url, data=data, files=files,
                    headers=headers
                )

            dip_response = DipResponse(
                data_idx=data_idx,
                status_code=raw_response.status_code,
                content=raw_response.content
            )

            data_point_retry_stat.status_code = raw_response.status_code

        except Exception as e:
            error = repr(e)
            dip_response = DipResponse(data_idx=data_idx, error=error)

            data_point_retry_stat.error = error

        data_point_retry_stat.end_time = datetime.now(UTC)
        return dip_response, data_point_retry_stat

    @staticmethod
    def __extract_files(data):
        files = {}

        if isinstance(data, (bytes, io.IOBase)):
            files = {'media': data}
            data = {}

        if isinstance(data, dict):
            keys_to_extract = []

            for key, value in data.items():
                if isinstance(value, (bytes, io.IOBase)):
                    keys_to_extract.append(key)

            for key in keys_to_extract:
                files[key] = data.pop(key)

        return data, files

    @staticmethod
    def __assign_predictions_from_dip_responses(predictions: list[bytes | None],
                                                dip_responses: list[DipResponse],
                                                ):
        for res in dip_responses:
            if res.status_code == 200:
                predictions[res.data_idx] = res.content

    @staticmethod
    def __assign_data_point_retry_stat_from_responses(predictions: list,
                                                      dip_responses: list[DipResponse]):
        for dip_res in dip_responses:
            if dip_res.status_code == 200:
                predictions[dip_res.data_idx] = dip_res.content
