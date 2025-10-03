import logging
from pymilvus import Collection, MilvusClient, CollectionSchema
from pymilvus.bulk_writer import RemoteBulkWriter, BulkFileType,bulk_import,LocalBulkWriter
import numpy as np
from app.utility import  client
import uuid
from app.config import settings
from app.utility import getSchema


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

connect_param =RemoteBulkWriter.AzureConnectParam(
            container_name=settings.AZURE_CONTAINER_NAME,
            conn_str=settings.AZURE_CONNECTION_STRING,
            account_url=settings.AZURE_ACCOUNT_URL
    )

async def write_and_upload_to_azure(data):
    # logger.info("Starting write and upload to MinIO...")
    # 1. Define schema
    # logger.info("Defining collection schema for Milvus...")
    collection = client.describe_collection(collection_name=settings.COLLECTION_NAME)
    # print(collection)

    # schema =getSchema()[0]
    # Verify schema
    # schema.verify()
    # logger.info(f"Collection schema defined.")
#     writer = LocalBulkWriter(
#     schema=schema,
#     local_path="./output",
#     chunk_size=512*1024*1024,
#     file_type=BulkFileType.JSON
# )
    try: 
        remote_path = str(uuid.uuid4()).replace("-", "") + "\\" + "data"
        logger.info(f"Initializing RemoteBulkWriter for Azure: {remote_path}")
        writer = RemoteBulkWriter(
        schema=CollectionSchema.construct_from_dict(collection),
        connect_param=connect_param,
        remote_path=remote_path,
        file_type=BulkFileType.PARQUET
    )
        # 4. Append rows one by one
        logger.info(f"Appending {len(data)} rows to the writer...")
        for idx, item in enumerate(data):
            try:
                writer.append_row(item)
            except Exception as e:
                logger.error(f"Error appending row {idx}: {e}")

        #for i in range(len(data["text"])):
        #    writer.append_row({
        #        "id": np.int64(i + 1),
        #        "text": data["text"][i],
        #    })
        logger.info("All rows appended to writer.")
        # 5. Commit once at the end
        writer.commit()
        logger.info(f"Commit successful. Files: {writer.batch_files}")
        return writer.batch_files
    except Exception as e:
        logger.error(f"Error in write_and_upload_to_azure: {e}")
        raise e
    
def bulk_import_from_azure(remote_paths):
    logger.info(f"Triggering bulk import in Milvus for collection:{remote_paths}")
    url = settings.MILVUS_HOST
    resp = bulk_import(
        url=url,
        collection_name=settings.COLLECTION_NAME,
        files=remote_paths)
    job_id = resp.json()['data']['jobId']
    logger.info(f"Bulk import started. Job ID: {job_id}")
    return job_id