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

# Room pricing configuration (per night in EUR)
ROOM_PRICING = {
    "Einzelzimmer": {"base_price": 89, "price_per_extra_guest": 0, "max_guests": 1},
    "DORMERO Zimmer": {"base_price": 119, "price_per_extra_guest": 25, "max_guests": 2},
    "Komfort Zimmer": {"base_price": 149, "price_per_extra_guest": 25, "max_guests": 2},
    "Junior Suite": {"base_price": 199, "price_per_extra_guest": 35, "max_guests": 4},
}

# Meeting room pricing (per day in EUR)
MEETING_ROOM_PRICING = {
    "Carolina Cik": {"half_day": 450, "full_day": 750, "price_per_person_catering": 35},
    "Carina Cörbchen": {"half_day": 280, "full_day": 450, "price_per_person_catering": 35},
}

