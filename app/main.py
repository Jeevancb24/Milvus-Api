import os
from fastapi import Body, FastAPI, UploadFile, File
import logging
from pymilvus.bulk_writer import list_import_jobs,get_import_progress
import json
from app.bulkImport import bulk_import_from_azure, write_and_upload_to_azure
from app.utility import  client, getSchema
from app.config import settings
from app.chunker import read_file_return_dict
import asyncio
from pymilvus import AnnSearchRequest, RRFRanker

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

    # logger.info(json.dumps(resp.json(), indent=4))
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

async def run_file_batch(file_list,batch_size):
    all_texts = []
    batch_count = 0
    failed_files = []
    current_file_batch = []
    for idx, file_path in enumerate(file_list):
        #Batch 10 files at a time!
        logger.info(f"[{idx+1}/{len(file_list)}] Reading file: {file_path}")
        data = await read_file_return_dict(file_path)
        current_file_batch.append(file_path)
        batch_count += 1
        if batch_count < batch_size and (idx + 1) < len(file_list):
            all_texts.extend(data)
            continue  # Continue accumulating files in the batch
        # Process the current batch
        # try catch in case of exception
        remote_paths = await write_and_upload_to_azure(all_texts)
    #     resp = bulk_import_from_azure(remote_paths)
    #     logger.info(f"JOB ID: {resp}")
    #     batch_count = 0
    #     all_texts = []  # Reset for next batch
    #     # Check for the progress of the job until it is completed
    #     while True:
    #         #check every 5 seconds
    #         await asyncio.sleep(5)
    #         progress = import_status(resp)
    #         if progress['data']['state'] in ['Completed', 'Failed']:
    #             logger.info(f"Import job {resp} finished with state: {progress['data']['state']}")
    #             if progress['data']['state'] == 'Failed':
    #                 failed_files.extend(current_file_batch)
    #             break
    #         else:
    #             logger.info(f"Import job is in state: {progress['data']['state']}.")
    #     current_file_batch = []  # Reset for next batch
    return remote_paths

@app.post("/bulk-import-folder/")
async def bulk_import_folder(folder_path: str = Body(..., embed=True)):
    logger.info(f"Bulk import from folder: {folder_path}")
    # Get all text files in the folder
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
    logger.info(f"Found {len(file_list)} text files in folder.")
    
    batch_size = 100  # Number of files to process in each batch
    failed_files = await run_file_batch(file_list,batch_size)

    # if failed_files:
    #     final_failed_files = await run_file_batch(failed_files,1)  # Process failed files one by one
    #     if final_failed_files:
    #         logger.error(f"Some files failed to import even after retrying: {final_failed_files}")
    # else:
    #     logger.info("All files processed and uploaded successfully!")
    
    #     #all_texts.append(data)
    # #logger.info("All files processed and uploaded!")
    #     # with open(file_path, 'r', encoding='utf-8') as f:
    #         # text = f.read()
    #         # all_texts.append(text)
    # # Prepare data for all files at once
    # # data = await prepareDataTxt(all_texts)
    
    # #logger.info(f"Bulk import completed. Response: {resp}")
    return {f"bulk_import_response:{failed_files}"}


@app.post("/batch-hybrid-query/")
async def batch_hybrid_query(
    queries_file: UploadFile = File(...),
    output_dir: str = Body("output", embed=True)
):
    """
    Accepts a JSON file with multiple queries:
    [
        {"query_num": "1", "query": "xxxx yyyy zzzz"},
        ...
    ]
    For each query, performs hybrid search and saves result as query_<query_num>.json in output_dir.
    """
    import os

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read and parse the uploaded JSON file
    content = await queries_file.read()
    queries = json.loads(content)

    results = []
    for q in queries:
        query_num = q.get("query_num")
        query_text = q.get("query")
        if not query_num or not query_text:
            continue

        # Prepare search requests
        search_param_1 = {
            "data": [query_text],
            "anns_field": "dense_vector",
            "param": {"nprobe": 10},
            "limit": 5
        }
        search_param_2 = {
            "data": [query_text],
            "anns_field": "sparse_vector",
            "param": {"drop_ratio_search": 0.2},
            "limit": 5
        }
        reqs = [AnnSearchRequest(**search_param_1), AnnSearchRequest(**search_param_2)]
        ranker = RRFRanker(100)

        # Perform hybrid search
        res = client.hybrid_search(
            collection_name=settings.COLLECTION_NAME,
            reqs=reqs,
            ranker=ranker,
            limit=5
        )

        # Extract top 5 doc names
        top_docs = []
        for hits in res:
            for hit in hits:
                doc_name = hit.get("doc_name") or hit.get("file_name") or str(hit.id)
                top_docs.append(doc_name)
            break

        # Prepare result
        result = {
            "query": query_text,
            "response": top_docs[:5]
        }

        # Save to file
        out_path = os.path.join(output_dir, f"query_{query_num}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        results.append({"query_num": query_num, "output_file": out_path})

    return {"status": "done", "results": results}

@app.post("/import-to-milvus/")
async def import_to_milvus():
    bulk_import_from_azure()
    return {"status": "Import triggered"}








    