"""
Query engine for hotel data retrieval.
"""
import json
from dataclasses import dataclass
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever

from rag.config import TOP_K_RESULTS, SIMILARITY_THRESHOLD, HOTEL_DATA_PATH
from rag.indexer import load_or_create_index, load_hotel_data


@dataclass
class QueryResult:
    """Result from a hotel data query."""
    question: str
    relevant_data: list[dict[str, Any]]
    source_texts: list[str]
    categories: list[str]
    has_relevant_data: bool
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "relevant_data": self.relevant_data,
            "source_texts": self.source_texts,
            "categories": self.categories,
            "has_relevant_data": self.has_relevant_data
        }


class HotelQueryEngine:
    """Engine for querying hotel data using semantic search."""
    
    def __init__(self, index: VectorStoreIndex | None = None):
        self.index = index or load_or_create_index()
        self.hotel_data = load_hotel_data()
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=TOP_K_RESULTS
        )
    
    def query(self, question: str) -> QueryResult:
        """
        Query the hotel data with a natural language question.
        Returns relevant data chunks based on semantic similarity.
        """
        # Retrieve relevant nodes
        nodes = self.retriever.retrieve(question)
        
        # Filter by similarity threshold
        relevant_nodes = [
            node for node in nodes 
            if node.score is not None and node.score >= SIMILARITY_THRESHOLD
        ]
        
        if not relevant_nodes:
            return QueryResult(
                question=question,
                relevant_data=[],
                source_texts=[],
                categories=[],
                has_relevant_data=False
            )
        
        # Extract data from relevant nodes
        source_texts = []
        categories = []
        relevant_data = []
        
        for node in relevant_nodes:
            source_texts.append(node.text)
            category = node.metadata.get("category", "unknown")
            if category not in categories:
                categories.append(category)
            
            # Get the structured data for this category
            category_data = self._get_category_data(category)
            if category_data:
                relevant_data.append({
                    "category": category,
                    "data": category_data,
                    "similarity_score": node.score
                })
        
        return QueryResult(
            question=question,
            relevant_data=relevant_data,
            source_texts=source_texts,
            categories=categories,
            has_relevant_data=True
        )
    
    def _get_category_data(self, category: str) -> dict[str, Any] | None:
        """Extract structured data for a specific category from hotel data."""
        category_mappings = {
            "basic_info": [
                "hotel_id", "name", "brand", "address", "postal_code", 
                "city", "country", "region", "description_short", "description_long"
            ],
            "contact": ["phone", "email", "website"],
            "parking": [
                "parking_spaces", "parking_fee_day", 
                "parking_height_limit_m", "parking_reservation_possible"
            ],
            "transportation": [
                "distance_airport_km", "driving_time_airport_min",
                "public_transport_description", "distance_city_center_min",
                "sbahn_line", "distance_sbahn_station_m", "bus_line_to_messe",
                "bus_stop_distance_m", "time_to_messe_public_min",
                "distance_messe_km", "time_to_messe_car_min"
            ],
            "rooms": [
                "total_rooms", "room_categories", "room_details",
                "non_smoking_hotel", "bed_options", "room_features"
            ],
            "bar": [
                "bar_name", "bar_description", "bar_opening_hours", "cashless_only"
            ],
            "fitness_wellness": [
                "fitness_available", "fitness_hours", "fitness_equipment",
                "sauna_available", "sauna_hours", "sauna_preheat_time_min",
                "wellness_area_description"
            ],
            "free_amenities": [
                "free_wifi", "free_cosmetics", "free_light_system",
                "free_sleeping_system", "free_minibar_day1", "free_fitness",
                "free_wellness", "free_late_checkout", "free_pets", "free_flatscreen_55"
            ],
            "meeting_rooms": [
                "meeting_rooms_total", "meeting_rooms",
                "event_team_available", "event_technical_equipment_available"
            ]
        }
        
        keys = category_mappings.get(category)
        if not keys:
            return None
        
        return {key: self.hotel_data.get(key) for key in keys if key in self.hotel_data}


# Singleton instance for reuse
_query_engine: HotelQueryEngine | None = None


def get_query_engine() -> HotelQueryEngine:
    """Get or create the query engine singleton."""
    global _query_engine
    if _query_engine is None:
        _query_engine = HotelQueryEngine()
    return _query_engine


def query_hotel_data(question: str) -> dict[str, Any]:
    """
    Convenience function to query hotel data.
    Returns a dictionary with the query results.
    """
    engine = get_query_engine()
    result = engine.query(question)
    return result.to_dict()

