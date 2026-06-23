from app.services import database
from app.services.faiss_vector_store import FAISSVectorStore

docs = database.list_documents()
print(f'Documents: {len(docs)}')
for d in docs:
    print(f'  {d["filename"]} - {d["chunk_count"]} chunks')

f = FAISSVectorStore()
s = f.get_stats()
print(f'Total FAISS chunks: {s["total_chunks"]}')
print(f'Embedding dimension: {s["dimension"]}')

stats = database.get_query_stats()
print(f'Total queries logged: {stats["total_queries"]}')
print(f'Average response time: {stats["average_response_time"]}s')
print(f'Unanswered queries: {stats["unanswered_queries"]}')
