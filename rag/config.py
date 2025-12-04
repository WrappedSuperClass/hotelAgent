import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
HOTEL_DATA_PATH = BASE_DIR / "hotel_data.json"
INDEX_STORAGE_PATH = BASE_DIR / "storage"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please create a .env file with your API key.")

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# Retrieval settings
TOP_K_RESULTS = 3
SIMILARITY_THRESHOLD = 0.3  

