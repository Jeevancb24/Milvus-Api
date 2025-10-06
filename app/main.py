import os
from fastapi import Body, FastAPI, UploadFile, File
from typing import List
import logging
from pymilvus.bulk_writer import list_import_jobs,get_import_progress
import json
from app.bulkImport import bulk_import_from_azure, write_and_upload_to_azure
from app.insertData import convert_bulk_data_to_row_dicts, insertData
from app.utility import  client, getSchema, prepareData, prepareDataTxt
from app.config import settings
from fastapi import Request
from pathlib import Path
from app.chunker import read_file_return_dict
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Bulk Import Service")

# -------------------
# API Routes
# -------------------
@app.post("/health/")
async def health_check():
    return {"status": "ok"}

@app.post("/create-collection/")
async def create_collection(collection_name):
    logger.info(f"Creating collection: {collection_name}")

    schema,index_params=getSchema()

    client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params
    )
    res = client.get_load_state(
    collection_name=collection_name
    )
    logger.info(f"Collection '{collection_name}' created with load state: {res}")
    return {"collection_name": collection_name, "load_state": res}

@app.get("/import-status/")
def import_status(job_id: str):
    resp = get_import_progress(
        url=settings.MILVUS_HOST,
        job_id=job_id,
    )

    logger.info(json.dumps(resp.json(), indent=4))
    return resp.json()

@app.get("/list-collections/")
async def list_collections():
    resp = list_import_jobs(
        url=settings.MILVUS_HOST,
        collection_name=settings.COLLECTION_NAME,
    )
    logger.info(json.dumps(resp.json(), indent=4))
    return resp.json()

@app.post("/drop-collection/")
async def drop_collection(collection_name):
    logger.info(f"Dropping collection:")
   
    client.drop_collection(
        collection_name=collection_name
    )
    logger.info(f"Collection '{collection_name}' dropped successfully.")
    return {"collection_name": collection_name, "status": "dropped"}

@app.post("/bulk-import-folder/")
async def bulk_import_folder(folder_path: str = Body(..., embed=True)):
    logger.info(f"Bulk import from folder: {folder_path}")
    # Get all text files in the folder
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
    logger.info(f"Found {len(file_list)} text files in folder.")
    #all_texts = []
    failed_files = []
    for idx, file_path in enumerate(file_list):
        logger.info(f"[{idx+1}/{len(file_list)}] Reading file: {file_path}")
        data = await read_file_return_dict(file_path)
        #all_texts.extend(data)
        # try catch in case of exception
        remote_paths = await write_and_upload_to_azure(data)
        resp = bulk_import_from_azure(remote_paths)
        logger.info(f"JOB ID: {resp}")
        # Check for the progress of the job until it is completed
        while True:
            progress = import_status(resp)
            if progress['data']['state'] in ['Completed', 'Failed']:
                logger.info(f"Import job {resp} finished with state: {progress['data']['state']}")
                if progress['data']['state'] == 'Failed':
                    failed_files.append(file_path)
                break
            else:
                logger.info(f"Import job is in state: {progress['data']['state']}.")
                #check every 5 seconds
                await asyncio.sleep(1)

    if failed_files:
        logger.error(f"Failed to import the following files: {failed_files}")
    else:
        logger.info("All files processed and uploaded successfully!")
    
        #all_texts.append(data)
    #logger.info("All files processed and uploaded!")
        # with open(file_path, 'r', encoding='utf-8') as f:
            # text = f.read()
            # all_texts.append(text)
    # Prepare data for all files at once
    # data = await prepareDataTxt(all_texts)
    
    #logger.info(f"Bulk import completed. Response: {resp}")
    return {"bulk_import_response"}

@app.post("/bulk-import/")
async def bulkImport(files: List[UploadFile] = File(...), request: Request = None):
    logger.info(f"Received {len(files)} files for import")
    all_remote_paths = []

    for idx, file in enumerate(files):
        logger.info(f"[{idx+1}/{len(files)}] Processing file: {file.filename}")
        pdf_path = Path(file.filename)
        with open(pdf_path, "wb") as f:
            f.write(await file.read())
            # Now pdf_path exists and can be read by PdfReader
        data=await prepareData(pdf_path)
        remote_paths = await write_and_upload_to_azure(data)
        all_remote_paths.extend(remote_paths)
        # cleanup
        pdf_path.unlink()  
    
    logger.info("All the files processed!")
    resp = bulk_import_from_azure(all_remote_paths)
    logger.info(f"Bulk import completed. Response: {resp}")
    return {"bulk_import_response": resp}

@app.post("/bulk-insert")
async def bulkInsert(folder_path: str = Body(..., embed=True)):
    logger.info(f"Bulk import from folder: {folder_path}")
    # Get all text files in the folder
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
    logger.info(f"Found {len(file_list)} text files in folder.")
    all_texts = []
    for idx, file_path in enumerate(file_list):
        logger.info(f"[{idx+1}/{len(file_list)}] Reading file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        all_texts.append(text)
    # Prepare data for all files at once
    data = await prepareDataTxt(all_texts)
    row_data = convert_bulk_data_to_row_dicts(data)
    res=await insertData(row_data)

    logger.info(f"All files processed and data inserted.{res}")
    return {"status": "All files processed and data inserted."}





    