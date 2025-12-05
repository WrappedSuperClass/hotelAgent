"""
FastAPI REST endpoint for hotel data RAG queries.
"""
import json
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.query_engine import get_query_engine, QueryResult
from rag.indexer import rebuild_index
from rag.config import ROOM_PRICING, MEETING_ROOM_PRICING


class RoomType(str, Enum):
    """Type of room to book."""
    HOTEL = "hotel"
    MEETING = "meeting"


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for querying hotel data."""
    question: str = Field(
        ..., 
        description="Natural language question about the hotel",
        min_length=1,
        max_length=500
    )


class QueryResponse(BaseModel):
    """Response model for hotel data queries."""
    question: str
    relevant_data: list[dict[str, Any]]
    source_texts: list[str]
    categories: list[str]
    has_relevant_data: bool


class RebuildResponse(BaseModel):
    """Response for index rebuild operation."""
    success: bool
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str


class BookingRequest(BaseModel):
    """Request model for room booking inquiry."""
    room_type: RoomType = Field(
        ...,
        description="Type of room: 'hotel' for accommodation or 'meeting' for conference rooms"
    )
    check_in: date = Field(
        ...,
        description="Check-in date (YYYY-MM-DD)"
    )
    check_out: date = Field(
        ...,
        description="Check-out date (YYYY-MM-DD)"
    )
    guests: int = Field(
        ...,
        ge=1,
        le=150,
        description="Number of guests/attendees"
    )
    include_catering: bool = Field(
        default=False,
        description="Include catering (only applicable for meeting rooms)"
    )


class RoomOption(BaseModel):
    """A single room option with pricing."""
    room_name: str
    room_category: str
    size_sqm: float | int | None = None
    max_capacity: int
    available_count: int | None = None
    price_per_night: float | None = None
    price_per_day: float | None = None
    total_price: float
    nights: int | None = None
    days: int | None = None
    features: list[str] = []
    notes: str | None = None


class BookingResponse(BaseModel):
    """Response model for booking inquiry."""
    room_type: str
    check_in: date
    check_out: date
    guests: int
    duration_nights: int | None = None
    duration_days: int | None = None
    available_options: list[RoomOption]
    hotel_name: str
    hotel_address: str
    contact_email: str
    contact_phone: str
    message: str


def _load_hotel_data() -> dict:
    """Load hotel data from JSON file."""
    data_path = Path(__file__).parent.parent / "hotel_data.json"
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_hotel_room_options(hotel_data: dict, guests: int, nights: int) -> list[RoomOption]:
    """Get available hotel room options based on guest count."""
    options = []
    room_details = hotel_data.get("room_details", {})
    room_features = hotel_data.get("room_features", [])
    
    for room_name, details in room_details.items():
        pricing = ROOM_PRICING.get(room_name)
        if not pricing:
            continue
            
        max_guests = pricing["max_guests"]
        if max_guests < guests:
            continue  # Room cannot accommodate this many guests
        
        # Calculate price
        base_price = pricing["base_price"]
        extra_guests = max(0, guests - 1) if max_guests > 1 else 0
        price_per_night = base_price + (extra_guests * pricing["price_per_extra_guest"])
        total_price = price_per_night * nights
        
        # Get room size
        size_sqm = details.get("sqm") or details.get("sqm_min")
        if details.get("sqm_max"):
            size_sqm = f"{details.get('sqm_min')}-{details.get('sqm_max')}"
        
        options.append(RoomOption(
            room_name=room_name,
            room_category="Hotel Room",
            size_sqm=details.get("sqm") or details.get("sqm_min"),
            max_capacity=max_guests,
            available_count=details.get("count"),
            price_per_night=price_per_night,
            total_price=total_price,
            nights=nights,
            features=room_features,
            notes=f"Size: {size_sqm} sqm" if size_sqm else None
        ))
    
    # Sort by price
    options.sort(key=lambda x: x.total_price)
    return options


def _get_meeting_room_options(
    hotel_data: dict, 
    guests: int, 
    days: int,
    include_catering: bool
) -> list[RoomOption]:
    """Get available meeting room options based on attendee count."""
    options = []
    meeting_rooms = hotel_data.get("meeting_rooms", {})
    
    for room_name, details in meeting_rooms.items():
        max_capacity = details.get("max_capacity", 0)
        if max_capacity < guests:
            continue  # Room cannot accommodate this many attendees
        
        pricing = MEETING_ROOM_PRICING.get(room_name)
        if not pricing:
            continue
        
        # Calculate price (using full day rate)
        price_per_day = pricing["full_day"]
        total_price = price_per_day * days
        
        # Add catering if requested
        catering_note = None
        if include_catering:
            catering_cost = pricing["price_per_person_catering"] * guests * days
            total_price += catering_cost
            catering_note = f"Includes catering: €{pricing['price_per_person_catering']}/person/day"
        
        features = []
        if details.get("daylight"):
            features.append("Natural daylight")
        if details.get("free_wifi"):
            features.append("Free WiFi")
        features.append(details.get("seating_options", "Flexible seating"))
        
        room_dimensions = f"{details.get('length_m', '?')}m × {details.get('width_m', '?')}m"
        
        options.append(RoomOption(
            room_name=room_name,
            room_category="Meeting Room",
            size_sqm=details.get("sqm"),
            max_capacity=max_capacity,
            price_per_day=price_per_day,
            total_price=total_price,
            days=days,
            features=features,
            notes=f"Dimensions: {room_dimensions}, Height: {details.get('height_m', '?')}m. {catering_note or ''}"
        ))
    
    # Sort by price
    options.sort(key=lambda x: x.total_price)
    return options


# Create FastAPI app
app = FastAPI(
    title="Hotel RAG API",
    description="RAG-powered API for querying hotel information",
    version="1.0.0"
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", service="hotel-rag")


@app.post("/query", response_model=QueryResponse)
async def query_hotel(request: QueryRequest):
    """
    Query hotel data with a natural language question.
    
    Returns relevant data from the hotel database based on semantic similarity.
    If no relevant data is found, has_relevant_data will be False.
    """
    try:
        engine = get_query_engine()
        result = engine.query(request.question)
        
        return QueryResponse(
            question=result.question,
            relevant_data=result.relevant_data,
            source_texts=result.source_texts,
            categories=result.categories,
            has_relevant_data=result.has_relevant_data
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Data file not found: {str(e)}")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Query failed: {str(e)}\n\nDetails:\n{error_details}"
        )


@app.post("/rebuild-index", response_model=RebuildResponse)
async def rebuild_hotel_index():
    """
    Rebuild the vector index from the hotel data file.
    
    Use this endpoint after updating the hotel_data.json file.
    """
    try:
        rebuild_index()
        # Reset the query engine to use the new index
        global _query_engine
        from rag import query_engine as qe_module
        qe_module._query_engine = None
        
        return RebuildResponse(success=True, message="Index rebuilt successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")


@app.post("/booking-request", response_model=BookingResponse)
async def request_booking(request: BookingRequest):
    """
    Request available rooms for booking.
    
    Provide check-in/check-out dates, number of guests, and room type
    to get a list of available options with pricing.
    
    - **room_type**: 'hotel' for accommodation or 'meeting' for conference rooms
    - **check_in**: Start date of the booking
    - **check_out**: End date of the booking
    - **guests**: Number of people (1-150)
    - **include_catering**: Add catering service (meeting rooms only)
    
    Returns available room options sorted by price with total cost calculation.
    """
    # Validate dates
    if request.check_out <= request.check_in:
        raise HTTPException(
            status_code=400,
            detail="Check-out date must be after check-in date"
        )
    
    # Calculate duration
    duration = (request.check_out - request.check_in).days
    
    try:
        hotel_data = _load_hotel_data()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Hotel data file not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid hotel data format")
    
    # Get available options based on room type
    if request.room_type == RoomType.HOTEL:
        options = _get_hotel_room_options(hotel_data, request.guests, duration)
        duration_label = "nights"
    else:
        options = _get_meeting_room_options(
            hotel_data, 
            request.guests, 
            duration,
            request.include_catering
        )
        duration_label = "days"
    
    # Build response message
    if options:
        message = f"Found {len(options)} available {request.room_type.value} room(s) for {request.guests} guest(s) over {duration} {duration_label}."
    else:
        message = f"No {request.room_type.value} rooms available for {request.guests} guest(s). Please contact us for group bookings or alternative arrangements."
    
    return BookingResponse(
        room_type=request.room_type.value,
        check_in=request.check_in,
        check_out=request.check_out,
        guests=request.guests,
        duration_nights=duration if request.room_type == RoomType.HOTEL else None,
        duration_days=duration if request.room_type == RoomType.MEETING else None,
        available_options=options,
        hotel_name=hotel_data.get("name", "DORMERO Hotel"),
        hotel_address=f"{hotel_data.get('address', '')}, {hotel_data.get('postal_code', '')} {hotel_data.get('city', '')}",
        contact_email=hotel_data.get("email", ""),
        contact_phone=hotel_data.get("phone", ""),
        message=message
    )

