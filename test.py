from src.medrag import MedRAG
import os
from src.adapt_medrag import AdaptiveMedRAG

os.environ["ARVAN_API_KEY"] = "d60b8d53-5f6c-5eb6-828a-031d503221d7"
os.environ["ARVAN_ENDPOINT"] = "https://arvancloudai.ir/gateway/models/Llama-3.3-70B-Instruct-R1AI/wXwicG2w6RhESLlIg7KlsetpzCShAX1k0bXHGZLX9nDHLxr4mJwXDMW-2Vunju0LPAXGRBNA8Hg4NhbIgiU-6UwAscgx9QDw1vINQTl1636Buf89X2lEaj3jhmNVJ9Q5qjRgH_G-9CxWta0kJqLV_LtR3XQypA0QnGRFDXtMfKtcE4Q0w5zta-2-Ul-b1rMjyoCaLU2oe4h7NM8-23TJZsUjYp9JrcetdbTBdzh2TDjNK1Iy0I10_c_8RQIN0B9LAEWZihx5ela0-k8AEHGCTwvU5zqzRXjNpwAdEQ/v1"


MODEL_NAME = "Arvan/Llama-3-3-70B-Instruct-R1AI"

adaptive_medrag = AdaptiveMedRAG(
    llm_name=MODEL_NAME,
    rag=True,
    follow_up=True,
    retriever_name="MedCPT",
    corpus_name="Textbooks",
    corpus_cache=True
)

retriever = adaptive_medrag.retrieval_system.retrievers[0][0]

target_id = "Gynecology_Novak_230"

retriever = adaptive_medrag.retrieval_system.retrievers[0][0]

found = False

for faiss_row, metadata in enumerate(retriever.metadatas):

    doc_id = f"{metadata['source']}_{metadata['index']}"

    if doc_id == target_id:

        embedding = retriever.index.reconstruct(faiss_row)

        print("Found document:", doc_id)
        print("FAISS row:", faiss_row)
        print("Embedding shape:", embedding.shape)
        print("First 5 values:", embedding[:5])

        found = True
        break

if not found:
    print("Document not found")