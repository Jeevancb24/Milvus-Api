import logging
from transformers import AutoTokenizer
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
import os

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
    logger.info(f"Reading text file: {file_path}")
    file_name = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        chunks = enforce_token_limit(text, MODEL_TOKEN_LIMIT, hf_tokenizer)
        data = [{"text": x, "file_name": file_name} for x in chunks]
        logger.info(f"File '{file_name}' read and split into {len(chunks)} chunks.")
    return data
