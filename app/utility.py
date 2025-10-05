import logging
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from pymilvus import MilvusClient, DataType,Function, FunctionType
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from app.config import settings
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_client():
    client = MilvusClient(
        uri=settings.MILVUS_HOST,
        token="root:Milvus"
    )
    return client

# Global Milvus client
client = get_client()

def extract_text_from_pdf(pdf_path):
    logger.info(f"Extracting text from PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        text += page_text + "\n" if page_text else ""
    logger.info(f"Total extracted text length: {len(text)} characters")
    return text

def split_text(text, chunk_size=512, overlap=256):
    logger.info(f"Splitting text into chunks (chunk_size={chunk_size}, overlap={overlap})")
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        # Ensure no chunk exceeds 2000 characters
        if len(chunk) <= 800:
            chunks.append(chunk)
        else:
            # If chunk is too long, split by character count
            for j in range(0, len(chunk), 800):
                sub_chunk = chunk[j:j+800]
                chunks.append(sub_chunk)
    logger.info(f"Total chunks created: {len(chunks)}")
    return chunks

# def generate_vectors(chunks,model_dense, model_sparse):
#     logger.info(f"Generating dense vectors for {len(chunks)} chunks")
#     dense_vecs = model_dense.encode(chunks).tolist()
#     logger.info(f"Dense vectors generated: {len(dense_vecs)}")

#     logger.info("Generating sparse vectors using TF-IDF")
#     X = model_sparse.fit_transform(chunks)
#     sparse_vecs = []
#     for i, row in enumerate(X):
#         coo = row.tocoo()
#         indices = coo.col.tolist()
#         values = coo.data.tolist()
#         logger.info(f"Chunk {i+1}: {len(indices)} nonzero sparse vector elements")
#         sparse_vecs.append({"indices": indices, "values": values})
#     logger.info(f"Sparse vectors generated: {len(sparse_vecs)}")
#     return dense_vecs, sparse_vecs

def getSchema():
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=6000,enable_analyzer=True)
    schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=2000,enable_analyzer=True)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=768)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    text_embedding_function = Function(
    name="tei_func",                            # Unique identifier for this embedding function
    function_type=FunctionType.TEXTEMBEDDING,   # Indicates a text embedding function
    input_field_names=["text"],             # Scalar field(s) containing text data to embed
    output_field_names=["dense_vector"],        # Vector field(s) for storing embeddings
    params={                                    # TEI specific parameters (function-level)
        "provider": "TEI",                      # Must be set to "TEI"
        "endpoint": "http://10.0.8.188:8080", # Required: Points to your TEI service address
        }
    )
    bm25_function = Function(
    name="text_bm25_emb", # Function name
    input_field_names=["text"], # Name of the VARCHAR field containing raw text data
    output_field_names=["sparse_vector"], # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
    function_type=FunctionType.BM25, # Set to `BM25`
    )
    schema.add_function(bm25_function)
    schema.add_function(text_embedding_function)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="id",
        index_type="AUTOINDEX"
    )

    index_params.add_index(
        field_name="dense_vector", 
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            "bm25_k1": 1.2,
            "bm25_b": 0.75
        }

    )
    return schema, index_params

async def prepareData(pdf_path):
    logger.info(f"File saved. Extracting text from PDF: {pdf_path}")
    text = extract_text_from_pdf(str(pdf_path))
    logger.info(f"Text extraction complete. Splitting text into chunks.")
    chunks = split_text(text)
    # logger.info(f"Text split into {len(chunks)} chunks. Generating vectors.")
    # dense_vecs, sparse_vecs = generate_vectors(chunks,model_dense=dense_model, model_sparse=sparse_model)

    data = {
        "text": chunks,
    }
    return data

async def prepareDataTxt(text):
    logger.info(f"Preparing data from text input.")
    if isinstance(text, list):
        # If input is a list, split each text and combine all chunks
        chunks = []
        for t in text:
            chunks.extend(split_text(t))
    else:
        chunks = split_text(text)
    data = {
        "text": chunks,
    }
    return data

#Initialize a global text splitter
splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=64, chunk_size=512)

async def read_file_return_dict(file_path):
    logger.info(f"Reading text file: {file_path}")
    #Read file name from path
    file_name = os.path.basename(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        chunks = splitter.split_text(text)
        # Convert the chunks into the required dictionary format
        # chunk text should be under key "text"
        data = [{"text": x, "file_name": file_name} for x in chunks]
        logger.info(f"File '{file_name}' read and split into {len(chunks)} chunks.")
    return data

    