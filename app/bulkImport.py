import logging
import uuid
from pymilvus import MilvusClient, DataType
from pymilvus.bulk_writer import RemoteBulkWriter, BulkFileType,bulk_import,LocalBulkWriter
import numpy as np

from app.config import settings
from app.utility import getSchema


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def write_and_upload_to_azure(data):
    logger.info("Starting write and upload to MinIO...")
    # 1. Define schema
    logger.info("Defining collection schema for Milvus...")

    schema =getSchema()[0]
     # Verify schema
    schema.verify()
    logger.info(f"Collection schema defined.")

    connect_param =RemoteBulkWriter.AzureConnectParam(
            container_name=settings.AZURE_CONTAINER_NAME,
            conn_str=settings.AZURE_CONNECTION_STRING,
            account_url=settings.AZURE_ACCOUNT_URL
    )
#     writer = LocalBulkWriter(
#     schema=schema,
#     local_path="./output",
#     chunk_size=512*1024*1024,
#     file_type=BulkFileType.JSON
# )
    writer = RemoteBulkWriter(
    schema=schema,
    connect_param=connect_param,
    remote_path="/",
    file_type=BulkFileType.PARQUET
)
    # 4. Append rows one by one
    logger.info(f"Appending {len(data['text'])} rows to the writer...")
    for i in range(len(data["text"])):
        writer.append_row({
            "id": np.int64(i + 1),
            "text": data["text"][i],
        })
    logger.info("All rows appended to writer.")
    # 5. Commit once at the end
    writer.commit()
    logger.info(f"Commit successful. Files: {writer.batch_files}")
    return writer.batch_files

def bulk_import_from_azure(remote_paths, collection_name):
    logger.info(f"Triggering bulk import in Milvus for collection: {collection_name}")
    url = settings.MILVUS_HOST
    resp = bulk_import(
        url=url,
        collection_name=collection_name,
        files=remote_paths)
    job_id = resp.json()['data']['jobId']
    logger.info(f"Bulk import started. Job ID: {job_id}")
    return job_id