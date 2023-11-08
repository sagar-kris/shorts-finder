import logging
import os
import time
import tiktoken
import openai
import pinecone

logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index("reels")

EMBEDDING_MODEL = "text-embedding-ada-002"
embedding_dimension = 1536
max_token_length_embedding = 8191
max_char_length_embedding = int(max_token_length_embedding * 2.5)

def current_milli_time():
    return round(time.time() * 1000)

def get_token_count(text: str, model_name: str="gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string."""
    # encoding = tiktoken.get_encoding(encoding_name)
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(text))
    logger.info(f'token count: {num_tokens}; token limit: {max_token_length_embedding}')
    return num_tokens

# chunk long sentences to stay within embedding model token limits
def chunk_embedding_data(string: str, n: int) -> list[str]:
    assert n >= 2
    string_list = []
    offset = int(max_char_length_embedding - (((max_char_length_embedding * n) - len(string))/(n - 1))) + 1
    logging.info(f'offset is {offset}')
    for idx in range(0, n):
        # print(f"inner loop count is {count}")
        starting_idx = idx * offset
        chunk = string[starting_idx : starting_idx + max_char_length_embedding]
        string_list.append(chunk)
    return string_list

def get_embedding_inner(client, string: str) -> list[float]:
    response = client.embeddings.create(
        input=string,
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding

def get_embedding(client, string: str) -> list[list[float]]:
    string_list = [string]
    token_count = get_token_count(string)
    if token_count >= max_token_length_embedding:
        num_chunks = int(token_count/max_token_length_embedding) + 1
        string_list = chunk_embedding_data(string, num_chunks)
    
    embeddings = []
    for string in string_list:
        embeddings.append(get_embedding_inner(client, string))
    return embeddings

# index health
def pinecone_index_health():
    index = pinecone.Index(pinecone.list_indexes()[0])
    index_stats_response = index.describe_index_stats()
    return index_stats_response

# push embeddings to pinecone
def pinecone_upsert(embeddings, namespace: str, id: str):
    logging.info(f'number of embeddings: {len(embeddings)}; namespace: {namespace}; id: {id}')
    vectors = [(str(int(current_milli_time())), list(embeddings[i]), {"id": id}) for i in range(len(embeddings))]
    # index = pinecone.Index(pinecone.list_indexes()[0])
    upsert_response = index.upsert(
        vectors=vectors,
        namespace=namespace
    )
    return upsert_response

# TODO: look into the `sparse_vector` param, could it be useful?
# get top k similar vectors to query
def pinecone_query(query_embedding, namespace: str, top_k: int=6):
    # index = pinecone.Index(pinecone.list_indexes()[0])
    query_response = index.query(
        vector=query_embedding,
        namespace=namespace,
        top_k=top_k,
        include_values=False,
        include_metadata=True,
    )
    return query_response

# get top k similar vectors to query with metadata filtering
def pinecone_query_filter(query_embedding, namespace: str, id: str, top_k: int=5):
    # index = pinecone.Index(pinecone.list_indexes()[0])
    query_response = index.query(
        vector=query_embedding,
        namespace=namespace,
        filter={
            "id": {"$in": [id]}
        },
        top_k=top_k,
        include_values=False,
        include_metadata=True,
    )
    return query_response
