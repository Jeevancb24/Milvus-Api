from app.config import settings
from app.utility import  get_client

async def insertData(data):
    client=get_client()
    res = client.insert(
        collection_name=settings.COLLECTION_NAME,
        data=data
    )
    return res

def convert_bulk_data_to_row_dicts(data: dict) -> list:
    """
    Convert bulk data dict with lists to a list of row dicts for Milvus insert.
    Expects keys: 'text', 'dense_vector', 'sparse_vector'.
    Returns: list of dicts, each dict is a row for Milvus insert.
    """
    texts = data.get("text", [])
    result = []
    for i in range(len(texts)):
        result.append({
            'id': i+1,
            'text': texts[i],
        })
    
    return result