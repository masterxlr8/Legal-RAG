from dotenv import load_dotenv
import os
load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
ollama_api_key = os.getenv("OLLAMA_API_KEY")
chunk_json_path = "./data/chunked_data/chunks_v2_2400docs.json"
embeddings_path = "./data/embeddings/embeddings_v2_2400docs.npy"
crossencoder_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
sentence_transformer_model_name = 'all-MiniLM-L6-v2'