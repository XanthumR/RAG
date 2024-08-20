import re
from langchain_community.embeddings.ollama import OllamaEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4

log_file_path = 'log.log'
log_lines = []

with open(log_file_path, 'r') as file:
    for line in file:
        log_lines.append(line)

log_pattern = re.compile(
    r'(?P<ip>\S+) \S+ \S+ \[(?P<time>[^\]]+)\] "(?P<method>\S+) (?P<url>\S+) \S+" (?P<status>\d{3}) (?P<size>\S+)')


def parse_log_line(line):
    match = log_pattern.match(line)
    if match:
        return match.groupdict()
    return None


ollama_emb = OllamaEmbeddings(
    model="llama2",
)

parsed_logs = [parse_log_line(line) for line in log_lines]
parsed_logs = [log for log in parsed_logs if log is not None]

index = faiss.IndexFlatL2(len(ollama_emb.embed_query(parsed_logs[0]['url'])),)
vector_store = FAISS(
    embedding_function=ollama_emb,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
documents = []
metadata = []
for log in parsed_logs:
    log1 = (log["url"].split("/"))[-1]
    if log1 != "":
        print(log1)
        vectors = ollama_emb.embed_query(log1.replace(".png","").replace(".jpg","").replace("-","").replace("_","").replace(".",""))
        documents.append((log1, vectors))
        metadata.append({"url": log1})


uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_embeddings(text_embeddings=documents, ids=uuids, metadatas=metadata)

vector_store.save_local("faiss_index")
