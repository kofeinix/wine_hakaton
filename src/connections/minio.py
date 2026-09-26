import io
import logging
import mimetypes
import ssl

from datetime import timedelta
from aiohttp import TCPConnector, ClientTimeout
from aiohttp_retry import ExponentialRetry, RetryClient
from miniopy_async import Minio
from miniopy_async.helpers import DictType
from aiohttp import ClientSession

from src.settings.settings import MinioSettings

logger = logging.getLogger(__name__)


class MinioObjectInfo:
    def __init__(
        self,
        object_name: str,
        size: int | None = None,
        content_type: str | None = None,
    ) -> None:
        self.object_name = object_name
        self.size = size
        self.content_type = content_type


class MinioClient:
    def __init__(self, minio_settings: MinioSettings):
        """
        Initializes the MinioClient class.

        This method sets up the instance variables for the Minio client, including the event loop for async operations.

        Args:
            minio_settings: configuration of Minio
        """
        self._config: MinioSettings = minio_settings
        self.storage: Minio | None = None
        self.endpoint: str = f"{minio_settings.host}:{minio_settings.port}"
        self.access_key: str = minio_settings.access_key
        self.secret_key: str = minio_settings.secret_key
        self.bucket: str = minio_settings.bucket
        self.verify_ssl: bool = minio_settings.verify_ssl
        self.ca_cert_path: str | None = minio_settings.ca_cert_path
        logger.info(
            f"MinioClient initialized, endpoint={self.endpoint}, bucket={self.bucket}, verify={self.verify_ssl}."
        )

    async def start(self):
        """
        Initialize the MinioClient instance.

        This method sets up the instance variables for the Minio client, including the event loop for async operations.
        """
        if self.storage is None:
            if self.verify_ssl:
                ssl_context = ssl.create_default_context()
                if self.ca_cert_path:
                    ssl_context.load_verify_locations(cafile=self.ca_cert_path)
            else:
                ssl_context = False

            timeout = timedelta(minutes=1).seconds
            retry_options = ExponentialRetry(
                attempts=2, factor=0.2, statuses={500, 502, 503, 504}
            )
            session = RetryClient(
                ClientSession(
                    connector=TCPConnector(limit=10, ssl=ssl_context),
                    timeout=ClientTimeout(connect=timeout, sock_read=timeout),
                ),
                retry_options=retry_options,
            )

            self.storage = Minio(
                endpoint=self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.verify_ssl,
                session=session,
            )
            assert self.storage is not None
            try:
                bucket_exists = await self.storage.bucket_exists(self.bucket)
                if not bucket_exists:
                    await self.storage.make_bucket(self.bucket)
            except Exception as e:
                raise e

            logger.info("MinioClient successfully started and ready for operations.")

    async def stop(self):
        """Run actions for stopping a service."""
        await self.storage.close_session()
        logger.info("MinioClient stopped.")

    async def put_data(
        self,
        filename: str,
        data: io.BytesIO,
        file_length: int = -1,
        metadata: DictType | None = None,
    ) -> None:
        """
        Write the provided bytes to a file in the storage, overwriting it if the file already exists.

        Args:
            filename: The name under which the file will be saved.
            data: The content to be written to the file.
            file_length: size of file.
            metadata: file metadata.
        """

        if self.storage is None:
            raise RuntimeError("MinioClient not started. Call start() first.")

        logger.debug(f"MinIO put to bucket {self.bucket} object {filename}")

        try:
            await self.storage.put_object(
                bucket_name=self.bucket,
                object_name=filename,
                data=data,
                length=file_length,
                metadata=metadata,
            )
        except Exception:
            logger.exception(
                f"Failed to put data {filename} to minio bucket {self.bucket}")
            raise

    async def put_file(
        self,
        filename: str,
        data: io.BytesIO,
        file_length: int = -1,
        metadata: DictType | None = None,
    ) -> str:
        """
        Upload a file to the storage.

        Args:
            filename: the name under which the file will be saved in the storage.
            data: the content to be written to the file.
            file_length: size of file.
            metadata: file metadata.

        Returns:
            str: the URL of the uploaded file.
        """
        if self.storage is None:
            raise RuntimeError("MinioClient not started. Call start() first.")

        if file_length == -1:
            file_length = data.getbuffer().nbytes

        logger.debug(f"MinIO put_file {filename} to bucket {self.bucket}")
        try:
            await self.storage.put_object(
                bucket_name=self.bucket,
                object_name=filename,
                data=data,
                length=file_length,
                metadata=metadata,
            )
        except Exception:
            logger.exception(f"Failed to put file {filename} to minio bucket {self.bucket}")
            raise

        url = f"{self.bucket}/{filename}"
        return url

    def parse_minio_link(self, minio_link: str) -> tuple[str, str]:
        """
        Parse minio_link to extract bucket and object_name.

        Args:
            minio_link: Link in format "bucket/object_name" or just "object_name"

        Returns:
            Tuple of (bucket, object_name)
        """
        if "/" in minio_link:
            parts = minio_link.split("/", 1)
            if parts[0] == self.bucket:
                bucket, obj = parts[0], parts[1]
            else:
                bucket, obj = self.bucket, minio_link
        else:
            bucket, obj = self.bucket, minio_link
        return bucket, obj

    async def get_file(self, filename: str) -> io.BytesIO | None:
        """
        Download the entire file from the storage.

        Args:
            filename: the name of the file it was saved with (can be "bucket/object_name" or just "object_name").

        Return:
            BytesIO | None : file content if present, None otherwise.
        """
        if self.storage is None:
            raise RuntimeError("MinioClient not started. Call start() first.")

        # Parse filename to get bucket and object_name
        bucket, object_name = self.parse_minio_link(filename)

        logger.debug(f"MinIO get object {object_name} from bucket {bucket}")

        try:
            response = await self.storage.get_object(
                bucket_name=bucket, object_name=object_name
            )
            file_data = io.BytesIO(await response.read())
            file_data.seek(0)
            response.close()
            response.release()
            return file_data
        except Exception:
            logger.exception(
                f"Exception occurred during downloading file {object_name} from the storage.",
            )
            return None

    async def get_file_with_content_type(self, filename: str) -> tuple[io.BytesIO, str] | None:
        if self.storage is None:
            raise RuntimeError("MinioClient not started. Call start() first.")

        bucket, object_name = self.parse_minio_link(filename)
        try:
            response = await self.storage.get_object(
                bucket_name=bucket,
                object_name=object_name,
            )
            header_content_type = response.headers.get("content-type")
            guessed_content_type = mimetypes.guess_type(object_name)[0]
            content_type = (
                guessed_content_type
                if header_content_type in {None, "application/octet-stream", "binary/octet-stream"}
                else header_content_type
            ) or "application/octet-stream"
            file_data = io.BytesIO(await response.read())
            file_data.seek(0)
            response.close()
            response.release()
            return file_data, content_type
        except Exception:
            logger.exception(
                "Exception occurred during downloading file %s from bucket %s.",
                object_name,
                bucket,
            )
            return None

    async def list_files(self, prefix: str, recursive: bool = True) -> list[MinioObjectInfo]:
        if self.storage is None:
            raise RuntimeError("MinioClient not started. Call start() first.")

        objects = self.storage.list_objects(
            bucket_name=self.bucket,
            prefix=prefix,
            recursive=recursive,
        )
        result: list[MinioObjectInfo] = []
        async for item in objects.gen_iterator():
            object_name = getattr(item, "object_name", None)
            if not object_name or object_name.endswith("/"):
                continue
            result.append(
                MinioObjectInfo(
                    object_name=object_name,
                    size=getattr(item, "size", None),
                    content_type=mimetypes.guess_type(object_name)[0],
                )
            )
        return result

    async def delete_file(self, filename: str) -> None:
        """
        Delete a file from storage.
        If the specified file does not exist, no action will be taken.

        Args:
            filename: the name of the file it was saved with.
        """
        if self.storage is None:
            raise RuntimeError("MinioClient not started. Call start() first.")

        try:
            await self.storage.remove_object(
                bucket_name=self.bucket,
                object_name=filename,
            )
        except Exception:
            logger.exception(
                f"Failed to delete file {filename} from minio bucket {self.bucket}",
            )
            raise
