import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from transformers import AutoTokenizer
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
import os
from pypdf import PdfReader
import io
import pandas as pd
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# choose a conservative max that is strictly less than model limit (allow margin)
MODEL_TOKEN_LIMIT = 384
SAFETY_MARGIN = 2
TOKENS_PER_CHUNK = MODEL_TOKEN_LIMIT - SAFETY_MARGIN  # e.g. 382

# create HF tokenizer matching the model so counts match the encoder used by TEI
hf_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2", use_fast=True)

# initialize token-aware splitter from the huggingface tokenizer
splitter = SentenceTransformersTokenTextSplitter.from_huggingface_tokenizer(
    tokenizer=hf_tokenizer,
    tokens_per_chunk=TOKENS_PER_CHUNK,
    chunk_overlap=0,
    model_name="sentence-transformers/all-mpnet-base-v2",
)

def enforce_token_limit(text: str, max_tokens: int, tokenizer) -> list[str]:
    """
    Ensure all returned chunks are <= max_tokens by checking token counts and
    re-splitting any chunks that still exceed the limit using a simple sliding
    window on sentences.
    """
    docs = splitter.split_text(text)
    safe_chunks = []
    for chunk in docs:
        token_count = splitter.count_tokens(text=chunk)
        if token_count <= max_tokens:
            safe_chunks.append(chunk)
            continue

        # fallback: split chunk by sentences and recompose to token-boundaries
        import re
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        current = []
        for s in sentences:
            candidate = " ".join(current + [s]) if current else s
            if splitter.count_tokens(text=candidate) <= max_tokens:
                current = candidate.split(" ")  # track tokens indirectly
                # store candidate as string for now
                current = [candidate]
            else:
                if current:
                    safe_chunks.append(current[0])
                # if single sentence already too big, hard truncate by tokens
                if splitter.count_tokens(text=s) > max_tokens:
                    # hard token-level truncate using tokenizer
                    toks = tokenizer.encode(s, add_special_tokens=False)
                    toks = toks[:max_tokens]
                    truncated = tokenizer.decode(toks, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    safe_chunks.append(truncated)
                    current = []
                else:
                    current = [s]
        if current:
            safe_chunks.append(current[0])

    return safe_chunks

async def read_file_return_dict(file_path):
    logger.info(f"Reading file: {file_path}")
    file_name = os.path.basename(file_path)
    file_extension = file_name.split('.')[-1].lower()

    # Read file content as bytes
    with open(file_path, "rb") as f:
        file_content = f.read()

    if file_extension == 'json':
        logger.info("Processing JSON file.")
        json_data = json.loads(file_content.decode("utf-8"))
        text_splitter = RecursiveJsonSplitter(
            max_chunk_size=2000  # You can adjust this as needed
        )
        chunks = text_splitter.split_json(json_data)
        # Convert each chunk (dict) to string for downstream processing
        chunks = [json.dumps(chunk, ensure_ascii=False) for chunk in chunks]
    else:
        logger.info(f"Processing {file_extension.upper()} file.")
        text = read_file(file_content, file_name)
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=20, length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        # chunks = enforce_token_limit(text, MODEL_TOKEN_LIMIT, hf_tokenizer)

    data = [{"text": x, "file_name": file_name} for x in chunks]
    logger.info(f"File '{file_name}' read and split into {len(chunks)} chunks.")
    return data

def read_file(file_content: bytes, file_name: str) -> str:
    # Extract file extension from the file name
    file_extension = file_name.split('.')[-1].lower()
    text = ""

    # Check the file type
    try:
        if file_extension == 'pdf':
            # Read PDF file
            pdf_reader = PdfReader(io.BytesIO(file_content))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Fallback to empty string if `extract_text` fails

        elif file_extension in ['csv', 'xlsx']:
            # Read CSV or Excel file
            if file_extension == 'csv':
                file_data = pd.read_csv(io.BytesIO(file_content))
            else:
                file_data = pd.read_excel(io.BytesIO(file_content), engine='openpyxl')

            # Combine text from all columns
            for column in file_data.select_dtypes(include=['object']).columns:
                text += " ".join(file_data[column].dropna().astype(str)) + " "

        elif file_extension == 'txt':
            # Read plain text file
            text = file_content.decode("utf-8")

        else:
            raise ValueError("Unsupported file type. Please upload a PDF, CSV, Excel, or text file.")

    except Exception as e:
        raise ValueError(f"Error processing file: {e}")

    if not text.strip():
        raise ValueError("The file appears to be empty or contains unsupported content.")

    return text