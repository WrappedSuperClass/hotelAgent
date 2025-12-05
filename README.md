# Hotel RAG API

A RAG-powered API for querying hotel information and making booking requests using natural language.

## Setup

1. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

3. Run the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

---

## Endpoints

### `GET /health`
Health check endpoint.

### `POST /query`
Query hotel information using natural language questions.

### `POST /rebuild-index`
Rebuild the vector index after updating hotel data.

### `POST /booking-request`
**Request available rooms using natural language.**

---

## Booking Request Endpoint

The `/booking-request` endpoint accepts natural language booking requests and returns available rooms with pricing.

### Required Information

Your booking request **must include** the following information:

| Information | Description | Examples |
|-------------|-------------|----------|
| **Room Type** | Whether you need a hotel room (for sleeping) or a meeting room (for conferences) | "hotel room", "accommodation", "stay", "meeting room", "conference room" |
| **Dates** | Check-in and check-out dates | "December 10th to 12th", "from January 5 to January 7", "next Monday for 2 nights" |
| **Number of Guests** | How many people will be staying or attending | "for 2 people", "3 guests", "30 attendees" |

### Optional Information

| Information | Description | When to Include |
|-------------|-------------|-----------------|
| **Catering** | Whether food service is needed | Only for meeting rooms: "with catering", "including lunch" |

### Example Requests

#### Hotel Room Bookings

```
"I need a hotel room for 2 people from December 10th to 12th"
```

```
"Looking for accommodation for 1 guest next Friday to Sunday"
```

```
"Book a room for my wife and me from January 15-17, 2025"
```

```
"I'd like to stay at your hotel for 3 nights starting December 20th, single occupancy"
```

#### Meeting Room Bookings

```
"Book a meeting room for 30 attendees on January 15th with catering"
```

```
"I need a conference room for a workshop with 20 people on March 5th"
```

```
"Reserve a meeting space for 50 people from February 10-11 including lunch"
```

```
"Looking for a presentation room for 15 attendees next Wednesday, no catering needed"
```

### Request Format

```json
{
  "request": "I need a hotel room for 2 people from December 10th to 12th"
}
```

### Response Format

The API returns available room options with:

- **Parsed booking details**: room type, dates, guest count
- **Available options**: list of rooms that can accommodate your request
- **Pricing**: per-night/per-day rates and total cost
- **Room features**: amenities, size, capacity
- **Hotel contact information**: for follow-up or confirmation

### Example Response

```json
{
  "original_request": "I need a hotel room for 2 people from December 10th to 12th",
  "room_type": "hotel",
  "check_in": "2025-12-10",
  "check_out": "2025-12-12",
  "guests": 2,
  "include_catering": false,
  "duration_nights": 2,
  "available_options": [
    {
      "room_name": "DORMERO Zimmer",
      "room_category": "Hotel Room",
      "size_sqm": 22,
      "max_capacity": 2,
      "available_count": 34,
      "price_per_night": 144,
      "total_price": 288,
      "nights": 2,
      "features": ["Badewanne oder Dusche", "Fön", "Kosmetikspiegel", "Schreibtisch", "Safe"],
      "notes": "Size: 22 sqm"
    }
  ],
  "hotel_name": "DORMERO Hotel München-Kirchheim Messe",
  "hotel_address": "Räterstraße 9, 85551 Kirchheim b. München",
  "contact_email": "kirchheim@dormero.de",
  "contact_phone": "+49 89 905040",
  "message": "Found 3 available hotel room(s) for 2 guest(s) over 2 nights."
}
```

---

## Pricing

### Hotel Rooms (per night)

| Room Type | Base Price | Extra Guest | Max Guests |
|-----------|------------|-------------|------------|
| Einzelzimmer | €89 | - | 1 |
| DORMERO Zimmer | €119 | +€25 | 2 |
| Komfort Zimmer | €149 | +€25 | 2 |
| Junior Suite | €199 | +€35 | 4 |

### Meeting Rooms (per day)

| Room | Full Day | Catering | Max Capacity |
|------|----------|----------|--------------|
| Carina Cörbchen | €450 | +€35/person | 50 |
| Carolina Cik | €750 | +€35/person | 120 |

---

## Configuration

Pricing and other settings can be configured in `rag/config.py`.

