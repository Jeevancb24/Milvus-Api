from azure.storage.blob import BlobServiceClient
import os
from app.config import settings
def upload_blob_file( file_path: str, blob_name: str ):
    connection_string = settings.AZURE_CONNECTION_STRING
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(settings.AZURE_CONTAINER_NAME)
    with open(file_path, "rb") as data:
        blob_client = container_client.upload_blob(name=blob_name, data=data, overwrite=True)

    return blob_client.url

