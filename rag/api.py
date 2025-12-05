"""
FastAPI REST endpoint for hotel data RAG queries.
"""
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from rag.query_engine import get_query_engine, QueryResult
from rag.indexer import rebuild_index
from rag.config import ROOM_PRICING, MEETING_ROOM_PRICING, OPENAI_API_KEY, INDEX_STORAGE_PATH


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
    """Request model for natural language booking inquiry."""
    request: str = Field(
        ...,
        description="Natural language booking request describing your needs",
        min_length=10,
        max_length=1000
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
    booking_id: str
    original_request: str
    room_type: str
    check_in: date
    check_out: date
    guests: int
    include_catering: bool = False
    duration_nights: int | None = None
    duration_days: int | None = None
    available_options: list[RoomOption]
    hotel_name: str
    hotel_address: str
    contact_email: str
    contact_phone: str
    message: str


class ParsedBookingDetails(BaseModel):
    """Parsed booking details from natural language."""
    room_type: str  # "hotel" or "meeting"
    check_in: str  # YYYY-MM-DD
    check_out: str  # YYYY-MM-DD
    guests: int
    include_catering: bool = False
    parsing_notes: str | None = None


def _parse_booking_request(natural_request: str) -> ParsedBookingDetails:
    """Use OpenAI to parse natural language booking request into structured data."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    system_prompt = f"""You are a hotel booking assistant. Extract booking details from the user's natural language request.
Today's date is {today}.

Extract the following information:
- room_type: "hotel" for accommodation/sleeping or "meeting" for conference/meeting rooms
- check_in: The start date in YYYY-MM-DD format
- check_out: The end date in YYYY-MM-DD format  
- guests: Number of people (default to 1 if not specified)
- include_catering: true if catering/food is mentioned for meeting rooms, false otherwise

IMPORTANT:
- If dates are relative (like "next Monday", "December 15th"), convert to YYYY-MM-DD
- If only one date is given, assume a 1-night stay for hotel or 1-day booking for meeting
- If year is not specified, assume the current or next occurrence of that date
- For meeting rooms, look for keywords like: meeting, conference, event, presentation, workshop
- For hotel rooms, look for keywords like: stay, sleep, overnight, accommodation, room

Respond with ONLY valid JSON in this exact format:
{{"room_type": "hotel", "check_in": "YYYY-MM-DD", "check_out": "YYYY-MM-DD", "guests": 1, "include_catering": false, "parsing_notes": "optional notes about assumptions made"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": natural_request}
        ],
        temperature=0.1,
        max_tokens=200
    )
    
    response_text = response.choices[0].message.content.strip()
    
    # Extract JSON from response (handle markdown code blocks)
    json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
    if not json_match:
        raise ValueError(f"Could not parse booking details from: {natural_request}")
    
    parsed = json.loads(json_match.group())
    return ParsedBookingDetails(**parsed)


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


# Booking storage functions
PENDING_BOOKINGS_PATH = INDEX_STORAGE_PATH / "pending_bookings.json"
CONFIRMED_BOOKINGS_PATH = INDEX_STORAGE_PATH / "confirmed_bookings.json"


def _load_pending_bookings() -> list[dict]:
    """Load pending bookings from storage."""
    if not PENDING_BOOKINGS_PATH.exists():
        return []
    try:
        with open(PENDING_BOOKINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def _save_pending_bookings(bookings: list[dict]) -> None:
    """Save pending bookings to storage."""
    PENDING_BOOKINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PENDING_BOOKINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(bookings, f, indent=2, default=str)


def _load_confirmed_bookings() -> list[dict]:
    """Load confirmed bookings from storage."""
    if not CONFIRMED_BOOKINGS_PATH.exists():
        return []
    try:
        with open(CONFIRMED_BOOKINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def _save_confirmed_bookings(bookings: list[dict]) -> None:
    """Save confirmed bookings to storage."""
    CONFIRMED_BOOKINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIRMED_BOOKINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(bookings, f, indent=2, default=str)


def _get_next_booking_id() -> str:
    """Generate next sequential booking ID (BK-001, BK-002, etc.)."""
    pending = _load_pending_bookings()
    confirmed = _load_confirmed_bookings()
    
    # Get all existing booking IDs
    all_ids = set()
    for booking in pending:
        all_ids.add(booking.get("booking_id", ""))
    for booking in confirmed:
        all_ids.add(booking.get("booking_id", ""))
    
    # Find next available ID
    counter = 1
    while True:
        booking_id = f"BK-{counter:03d}"
        if booking_id not in all_ids:
            return booking_id
        counter += 1


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
    Request available rooms for booking using natural language.
    
    Send a natural language request describing your booking needs.
    The system will parse your request and return available options with pricing.
    
    **Required information in your request:**
    - Type of room: hotel room (for sleeping) or meeting room (for conferences)
    - Date(s): when you want to check in and check out
    - Number of guests/attendees
    - For meeting rooms: whether you need catering (optional)
    
    **Example requests:**
    - "I need a hotel room for 2 people from December 10th to 12th"
    - "Book a meeting room for 30 attendees on January 15th with catering"
    - "Looking for accommodation for 1 guest next Friday to Sunday"
    
    Returns available room options sorted by price with total cost calculation.
    """
    # Parse natural language request
    try:
        parsed = _parse_booking_request(request.request)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not understand booking request. Please include: room type (hotel/meeting), dates, and number of guests. Error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing request: {str(e)}"
        )
    
    # Convert parsed dates to date objects
    try:
        check_in = datetime.strptime(parsed.check_in, "%Y-%m-%d").date()
        check_out = datetime.strptime(parsed.check_out, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Could not parse dates from your request. Please specify clear dates."
        )
    
    # Validate dates
    if check_out <= check_in:
        raise HTTPException(
            status_code=400,
            detail="Check-out date must be after check-in date"
        )
    
    # Validate guests
    guests = parsed.guests
    if guests < 1 or guests > 150:
        raise HTTPException(
            status_code=400,
            detail="Number of guests must be between 1 and 150"
        )
    
    # Calculate duration
    duration = (check_out - check_in).days
    
    try:
        hotel_data = _load_hotel_data()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Hotel data file not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid hotel data format")
    
    # Determine room type
    room_type = parsed.room_type.lower()
    
    # Get available options based on room type
    if room_type == "hotel":
        options = _get_hotel_room_options(hotel_data, guests, duration)
        duration_label = "nights"
    else:
        options = _get_meeting_room_options(
            hotel_data, 
            guests, 
            duration,
            parsed.include_catering
        )
        duration_label = "days"
    
    # Build response message
    if options:
        message = f"Found {len(options)} available {room_type} room(s) for {guests} guest(s) over {duration} {duration_label}."
        if parsed.parsing_notes:
            message += f" Note: {parsed.parsing_notes}"
    else:
        message = f"No {room_type} rooms available for {guests} guest(s). Please contact us for group bookings or alternative arrangements."
    
    # Generate booking ID
    booking_id = _get_next_booking_id()
    
    # Create booking response
    booking_response = BookingResponse(
        booking_id=booking_id,
        original_request=request.request,
        room_type=room_type,
        check_in=check_in,
        check_out=check_out,
        guests=guests,
        include_catering=parsed.include_catering,
        duration_nights=duration if room_type == "hotel" else None,
        duration_days=duration if room_type == "meeting" else None,
        available_options=options,
        hotel_name=hotel_data.get("name", "DORMERO Hotel"),
        hotel_address=f"{hotel_data.get('address', '')}, {hotel_data.get('postal_code', '')} {hotel_data.get('city', '')}",
        contact_email=hotel_data.get("email", ""),
        contact_phone=hotel_data.get("phone", ""),
        message=message
    )
    
    # Store pending booking
    pending_booking = {
        "booking_id": booking_id,
        "original_request": request.request,
        "room_type": room_type,
        "check_in": check_in.isoformat(),
        "check_out": check_out.isoformat(),
        "guests": guests,
        "include_catering": parsed.include_catering,
        "duration_nights": duration if room_type == "hotel" else None,
        "duration_days": duration if room_type == "meeting" else None,
        "available_options": [option.model_dump() for option in options],
        "hotel_name": hotel_data.get("name", "DORMERO Hotel"),
        "hotel_address": f"{hotel_data.get('address', '')}, {hotel_data.get('postal_code', '')} {hotel_data.get('city', '')}",
        "contact_email": hotel_data.get("email", ""),
        "contact_phone": hotel_data.get("phone", ""),
        "timestamp": datetime.now().isoformat()
    }
    
    # Load existing pending bookings and add new one
    pending_bookings = _load_pending_bookings()
    pending_bookings.append(pending_booking)
    _save_pending_bookings(pending_bookings)
    
    return booking_response


class ConfirmBookingRequest(BaseModel):
    """Request model for confirming a booking."""
    booking_id: str = Field(..., description="Booking ID to confirm")
    room_name: str = Field(..., description="Name of the room option to confirm")


class ConfirmBookingResponse(BaseModel):
    """Response model for booking confirmation."""
    success: bool
    booking_id: str
    message: str
    confirmed_booking: dict[str, Any]


@app.post("/confirm-booking", response_model=ConfirmBookingResponse)
async def confirm_booking(request: ConfirmBookingRequest):
    """
    Confirm a booking request by selecting a specific room option.
    
    Requires:
    - **booking_id**: The booking ID returned from /booking-request
    - **room_name**: The name of the room option to confirm (must match one of the available_options)
    
    The booking will be moved from pending to confirmed status.
    """
    # Check if booking is already confirmed
    confirmed_bookings = _load_confirmed_bookings()
    for confirmed in confirmed_bookings:
        if confirmed.get("booking_id") == request.booking_id:
            raise HTTPException(
                status_code=400,
                detail="Booking already confirmed"
            )
    
    # Load pending bookings
    pending_bookings = _load_pending_bookings()
    pending_booking = None
    for booking in pending_bookings:
        if booking.get("booking_id") == request.booking_id:
            pending_booking = booking
            break
    
    if not pending_booking:
        raise HTTPException(
            status_code=400,
            detail="Booking ID not found"
        )
    
    # Find the selected room option
    available_options = pending_booking.get("available_options", [])
    selected_room = None
    for option in available_options:
        if option.get("room_name") == request.room_name:
            selected_room = option
            break
    
    if not selected_room:
        raise HTTPException(
            status_code=400,
            detail="Room option not available for this booking"
        )
    
    # Create confirmed booking
    confirmed_booking = {
        "booking_id": request.booking_id,
        "original_request": pending_booking.get("original_request"),
        "room_type": pending_booking.get("room_type"),
        "check_in": pending_booking.get("check_in"),
        "check_out": pending_booking.get("check_out"),
        "guests": pending_booking.get("guests"),
        "include_catering": pending_booking.get("include_catering", False),
        "duration_nights": pending_booking.get("duration_nights"),
        "duration_days": pending_booking.get("duration_days"),
        "selected_room": selected_room,
        "hotel_name": pending_booking.get("hotel_name"),
        "hotel_address": pending_booking.get("hotel_address"),
        "contact_email": pending_booking.get("contact_email"),
        "contact_phone": pending_booking.get("contact_phone"),
        "confirmation_timestamp": datetime.now().isoformat()
    }
    
    # Remove from pending and add to confirmed
    pending_bookings = [b for b in pending_bookings if b.get("booking_id") != request.booking_id]
    _save_pending_bookings(pending_bookings)
    
    confirmed_bookings.append(confirmed_booking)
    _save_confirmed_bookings(confirmed_bookings)
    
    return ConfirmBookingResponse(
        success=True,
        booking_id=request.booking_id,
        message=f"Booking {request.booking_id} confirmed for {selected_room.get('room_name')}",
        confirmed_booking=confirmed_booking
    )


@app.get("/confirmed-bookings", response_model=list[dict[str, Any]])
async def get_confirmed_bookings():
    """
    Get all confirmed bookings.
    
    Returns a list of all confirmed bookings with their details and selected room options.
    """
    confirmed_bookings = _load_confirmed_bookings()
    return confirmed_bookings

