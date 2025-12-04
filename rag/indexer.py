"""
Hotel data indexer - chunks and embeds hotel JSON data for retrieval.
"""
import json
from pathlib import Path
from typing import Any

from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

from rag.config import HOTEL_DATA_PATH, INDEX_STORAGE_PATH, EMBEDDING_MODEL, OPENAI_API_KEY


def load_hotel_data(path: Path = HOTEL_DATA_PATH) -> dict[str, Any]:
    """Load hotel data from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Hotel data file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            raise ValueError(f"Hotel data file is empty: {path}")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in hotel data file: {e}")


def create_semantic_chunks(hotel_data: dict[str, Any]) -> list[Document]:
    """
    Create semantically meaningful document chunks from hotel data.
    Each chunk represents a distinct topic/section for better retrieval.
    """
    documents = []
    
    # Basic hotel information
    basic_info = f"""
Hotel Name: {hotel_data.get('name', 'N/A')}
Brand: {hotel_data.get('brand', 'N/A')}
Address: {hotel_data.get('address', 'N/A')}, {hotel_data.get('postal_code', 'N/A')} {hotel_data.get('city', 'N/A')}
Country: {hotel_data.get('country', 'N/A')}
Region: {hotel_data.get('region', 'N/A')}
Short Description: {hotel_data.get('description_short', 'N/A')}
Long Description: {hotel_data.get('description_long', 'N/A')}
    """.strip()
    documents.append(Document(
        text=basic_info,
        metadata={"category": "basic_info", "hotel_id": hotel_data.get("hotel_id")}
    ))
    
    # Contact information
    contact_info = f"""
Hotel Contact Information:
Phone: {hotel_data.get('phone', 'N/A')}
Email: {hotel_data.get('email', 'N/A')}
Website: {hotel_data.get('website', 'N/A')}
    """.strip()
    documents.append(Document(
        text=contact_info,
        metadata={"category": "contact", "hotel_id": hotel_data.get("hotel_id")}
    ))
    
    # Parking information
    parking_info = f"""
Parking Information:
Number of parking spaces: {hotel_data.get('parking_spaces', 'N/A')}
Parking fee per day: {hotel_data.get('parking_fee_day', 'N/A')} EUR
Height limit: {hotel_data.get('parking_height_limit_m', 'N/A')} meters
Reservation possible: {'Yes' if hotel_data.get('parking_reservation_possible') else 'No'}
    """.strip()
    documents.append(Document(
        text=parking_info,
        metadata={"category": "parking", "hotel_id": hotel_data.get("hotel_id")}
    ))
    
    # Transportation and location
    transport_info = f"""
Transportation and Location:
Distance to airport: {hotel_data.get('distance_airport_km', 'N/A')} km
Driving time to airport: {hotel_data.get('driving_time_airport_min', 'N/A')} minutes
Public transport: {hotel_data.get('public_transport_description', 'N/A')}
Distance to city center: {hotel_data.get('distance_city_center_min', 'N/A')} minutes
S-Bahn line: {hotel_data.get('sbahn_line', 'N/A')}
Bus line to Messe: {hotel_data.get('bus_line_to_messe', 'N/A')}
Bus stop distance: {hotel_data.get('bus_stop_distance_m', 'N/A')} meters
Time to Messe by public transport: {hotel_data.get('time_to_messe_public_min', 'N/A')} minutes
Distance to Messe: {hotel_data.get('distance_messe_km', 'N/A')} km
Time to Messe by car: {hotel_data.get('time_to_messe_car_min', 'N/A')} minutes
    """.strip()
    documents.append(Document(
        text=transport_info,
        metadata={"category": "transportation", "hotel_id": hotel_data.get("hotel_id")}
    ))
    
    # Room information
    room_categories = hotel_data.get('room_categories', [])
    room_details = hotel_data.get('room_details', {})
    room_features = hotel_data.get('room_features', [])
    bed_options = hotel_data.get('bed_options', [])
    
    room_details_text = "\n".join([
        f"- {name}: {details.get('sqm', details.get('sqm_min', 'N/A'))} sqm, {details.get('count', 'N/A')} rooms available"
        for name, details in room_details.items()
    ])
    
    room_info = f"""
Room Information:
Total rooms: {hotel_data.get('total_rooms', 'N/A')}
Non-smoking hotel: {'Yes' if hotel_data.get('non_smoking_hotel') else 'No'}
Room categories: {', '.join(room_categories)}
Room details:
{room_details_text}
Bed options: {', '.join(bed_options)}
Room features: {', '.join(room_features)}
    """.strip()
    documents.append(Document(
        text=room_info,
        metadata={"category": "rooms", "hotel_id": hotel_data.get("hotel_id")}
    ))
    
    # Bar information
    bar_info = f"""
Bar Information:
Bar name: {hotel_data.get('bar_name', 'N/A')}
Description: {hotel_data.get('bar_description', 'N/A')}
Opening hours: {hotel_data.get('bar_opening_hours', 'N/A')}
Cashless only: {'Yes' if hotel_data.get('cashless_only') else 'No'}
    """.strip()
    documents.append(Document(
        text=bar_info,
        metadata={"category": "bar", "hotel_id": hotel_data.get("hotel_id")}
    ))
    
    # Fitness and wellness
    fitness_equipment = hotel_data.get('fitness_equipment', [])
    wellness_info = f"""
Fitness and Wellness:
Fitness available: {'Yes' if hotel_data.get('fitness_available') else 'No'}
Fitness hours: {hotel_data.get('fitness_hours', 'N/A')}
Fitness equipment: {', '.join(fitness_equipment)}
Sauna available: {'Yes' if hotel_data.get('sauna_available') else 'No'}
Sauna hours: {hotel_data.get('sauna_hours', 'N/A')}
Sauna preheat time: {hotel_data.get('sauna_preheat_time_min', 'N/A')} minutes
Wellness area: {hotel_data.get('wellness_area_description', 'N/A')}
    """.strip()
    documents.append(Document(
        text=wellness_info,
        metadata={"category": "fitness_wellness", "hotel_id": hotel_data.get("hotel_id")}
    ))
    
    # Free amenities
    free_amenities = f"""
Free Amenities and Services:
Free WiFi: {'Yes' if hotel_data.get('free_wifi') else 'No'}
Free cosmetics: {'Yes' if hotel_data.get('free_cosmetics') else 'No'}
Free light system: {'Yes' if hotel_data.get('free_light_system') else 'No'}
Free sleeping system: {'Yes' if hotel_data.get('free_sleeping_system') else 'No'}
Free minibar on first day: {'Yes' if hotel_data.get('free_minibar_day1') else 'No'}
Free fitness access: {'Yes' if hotel_data.get('free_fitness') else 'No'}
Free wellness access: {'Yes' if hotel_data.get('free_wellness') else 'No'}
Free late checkout: {hotel_data.get('free_late_checkout', 'N/A')}
Free pets allowed: {'Yes' if hotel_data.get('free_pets') else 'No'}
Free 55" flatscreen TV: {'Yes' if hotel_data.get('free_flatscreen_55') else 'No'}
    """.strip()
    documents.append(Document(
        text=free_amenities,
        metadata={"category": "free_amenities", "hotel_id": hotel_data.get("hotel_id")}
    ))
    
    # Meeting rooms
    meeting_rooms = hotel_data.get('meeting_rooms', {})
    meeting_rooms_text = ""
    for room_name, details in meeting_rooms.items():
        meeting_rooms_text += f"""
Room: {room_name}
  - Size: {details.get('sqm', 'N/A')} sqm
  - Height: {details.get('height_m', 'N/A')} m
  - Dimensions: {details.get('length_m', 'N/A')}m x {details.get('width_m', 'N/A')}m
  - Daylight: {'Yes' if details.get('daylight') else 'No'}
  - Maximum capacity: {details.get('max_capacity', 'N/A')} people
  - Seating options: {details.get('seating_options', 'N/A')}
  - WiFi: {'Yes' if details.get('free_wifi') else 'No'}
"""
    
    meeting_info = f"""
Meeting Rooms and Events:
Total meeting rooms: {hotel_data.get('meeting_rooms_total', 'N/A')}
Event team available: {'Yes' if hotel_data.get('event_team_available') else 'No'}
Technical equipment: {hotel_data.get('event_technical_equipment_available', 'N/A')}

Meeting Room Details:
{meeting_rooms_text}
    """.strip()
    documents.append(Document(
        text=meeting_info,
        metadata={"category": "meeting_rooms", "hotel_id": hotel_data.get("hotel_id")}
    ))
    
    return documents


def create_index(documents: list[Document], persist: bool = True) -> VectorStoreIndex:
    """Create a vector store index from documents."""
    embed_model = OpenAIEmbedding(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )
    
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
    )
    
    if persist:
        INDEX_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(INDEX_STORAGE_PATH))
    
    return index


def load_or_create_index() -> VectorStoreIndex:
    """Load existing index from storage or create a new one."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required but not set")
    
    embed_model = OpenAIEmbedding(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )
    
    if INDEX_STORAGE_PATH.exists():
        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_STORAGE_PATH))
            return load_index_from_storage(storage_context, embed_model=embed_model)
        except Exception as e:
            # If loading fails, delete corrupted storage and rebuild
            import shutil
            print(f"Warning: Failed to load index from storage: {e}")
            print("Rebuilding index...")
            if INDEX_STORAGE_PATH.exists():
                shutil.rmtree(INDEX_STORAGE_PATH)
    
    # Create new index
    hotel_data = load_hotel_data()
    documents = create_semantic_chunks(hotel_data)
    return create_index(documents, persist=True)


def rebuild_index() -> VectorStoreIndex:
    """Force rebuild the index from hotel data."""
    hotel_data = load_hotel_data()
    documents = create_semantic_chunks(hotel_data)
    return create_index(documents, persist=True)

