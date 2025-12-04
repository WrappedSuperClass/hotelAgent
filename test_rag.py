#!/usr/bin/env python3
"""
Comprehensive RAG Test Suite for Hotel Data
============================================
Tests various question types and validates retrieval accuracy.
"""
import requests
import json
from dataclasses import dataclass
from typing import Optional

API_URL = "http://localhost:8000/query"

# ═══════════════════════════════════════════════════════════════════════════════
# Test Case Definitions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestCase:
    question: str
    expected_categories: list[str]  # Expected category matches
    should_have_data: bool  # Whether we expect relevant data
    description: str  # What we're testing


TEST_CASES = [
    # ─────────────────────────────────────────────────────────────────────────────
    # PARKING QUESTIONS
    # ─────────────────────────────────────────────────────────────────────────────
    TestCase(
        question="What are the parking fees?",
        expected_categories=["parking"],
        should_have_data=True,
        description="Direct parking fee inquiry"
    ),
    TestCase(
        question="How many parking spaces are available?",
        expected_categories=["parking"],
        should_have_data=True,
        description="Parking capacity question"
    ),
    TestCase(
        question="Can I reserve a parking spot?",
        expected_categories=["parking"],
        should_have_data=True,
        description="Parking reservation inquiry"
    ),
    TestCase(
        question="What is the height limit for the parking garage?",
        expected_categories=["parking"],
        should_have_data=True,
        description="Parking height restriction"
    ),
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ROOM QUESTIONS
    # ─────────────────────────────────────────────────────────────────────────────
    TestCase(
        question="How many rooms does the hotel have?",
        expected_categories=["rooms"],
        should_have_data=True,
        description="Total room count"
    ),
    TestCase(
        question="What room types are available?",
        expected_categories=["rooms"],
        should_have_data=True,
        description="Room categories inquiry"
    ),
    TestCase(
        question="Do you have suites?",
        expected_categories=["rooms"],
        should_have_data=True,
        description="Suite availability"
    ),
    TestCase(
        question="What bed sizes do you offer?",
        expected_categories=["rooms"],
        should_have_data=True,
        description="Bed options inquiry"
    ),
    TestCase(
        question="Is smoking allowed in rooms?",
        expected_categories=["rooms"],
        should_have_data=True,
        description="Smoking policy"
    ),
    TestCase(
        question="What amenities are in the room?",
        expected_categories=["rooms"],
        should_have_data=True,
        description="Room features/amenities"
    ),
    
    # ─────────────────────────────────────────────────────────────────────────────
    # FITNESS & WELLNESS QUESTIONS
    # ─────────────────────────────────────────────────────────────────────────────
    TestCase(
        question="Is there a gym?",
        expected_categories=["fitness_wellness"],
        should_have_data=True,
        description="Gym availability"
    ),
    TestCase(
        question="What are the fitness center hours?",
        expected_categories=["fitness_wellness"],
        should_have_data=True,
        description="Fitness hours inquiry"
    ),
    TestCase(
        question="Do you have a sauna?",
        expected_categories=["fitness_wellness"],
        should_have_data=True,
        description="Sauna availability"
    ),
    TestCase(
        question="What gym equipment is available?",
        expected_categories=["fitness_wellness"],
        should_have_data=True,
        description="Fitness equipment list"
    ),
    TestCase(
        question="Is there a spa or wellness area?",
        expected_categories=["fitness_wellness"],
        should_have_data=True,
        description="Wellness facilities"
    ),
    
    # ─────────────────────────────────────────────────────────────────────────────
    # BAR & DINING QUESTIONS
    # ─────────────────────────────────────────────────────────────────────────────
    TestCase(
        question="Is there a bar in the hotel?",
        expected_categories=["bar"],
        should_have_data=True,
        description="Bar availability"
    ),
    TestCase(
        question="What are the bar opening hours?",
        expected_categories=["bar"],
        should_have_data=True,
        description="Bar hours inquiry"
    ),
    TestCase(
        question="Can I pay with cash at the bar?",
        expected_categories=["bar"],
        should_have_data=True,
        description="Payment methods"
    ),
    
    # ─────────────────────────────────────────────────────────────────────────────
    # TRANSPORTATION & LOCATION QUESTIONS
    # ─────────────────────────────────────────────────────────────────────────────
    TestCase(
        question="How far is the airport?",
        expected_categories=["transportation"],
        should_have_data=True,
        description="Airport distance"
    ),
    TestCase(
        question="How do I get to the Messe?",
        expected_categories=["transportation"],
        should_have_data=True,
        description="Messe directions"
    ),
    TestCase(
        question="What public transport is nearby?",
        expected_categories=["transportation"],
        should_have_data=True,
        description="Public transport info"
    ),
    TestCase(
        question="How long to reach city center?",
        expected_categories=["transportation"],
        should_have_data=True,
        description="City center distance"
    ),
    TestCase(
        question="Which S-Bahn line is closest?",
        expected_categories=["transportation"],
        should_have_data=True,
        description="S-Bahn information"
    ),
    
    # ─────────────────────────────────────────────────────────────────────────────
    # FREE AMENITIES QUESTIONS
    # ─────────────────────────────────────────────────────────────────────────────
    TestCase(
        question="Is WiFi free?",
        expected_categories=["free_amenities"],
        should_have_data=True,
        description="WiFi availability"
    ),
    TestCase(
        question="Are pets allowed?",
        expected_categories=["free_amenities"],
        should_have_data=True,
        description="Pet policy"
    ),
    TestCase(
        question="Is the minibar free?",
        expected_categories=["free_amenities"],
        should_have_data=True,
        description="Minibar policy"
    ),
    TestCase(
        question="What time is late checkout?",
        expected_categories=["free_amenities"],
        should_have_data=True,
        description="Late checkout policy"
    ),
    
    # ─────────────────────────────────────────────────────────────────────────────
    # MEETING ROOMS QUESTIONS
    # ─────────────────────────────────────────────────────────────────────────────
    TestCase(
        question="Do you have conference rooms?",
        expected_categories=["meeting_rooms"],
        should_have_data=True,
        description="Conference room availability"
    ),
    TestCase(
        question="What is the largest meeting room capacity?",
        expected_categories=["meeting_rooms"],
        should_have_data=True,
        description="Meeting room capacity"
    ),
    TestCase(
        question="Can I host an event at the hotel?",
        expected_categories=["meeting_rooms"],
        should_have_data=True,
        description="Event hosting"
    ),
    
    # ─────────────────────────────────────────────────────────────────────────────
    # CONTACT & BASIC INFO QUESTIONS
    # ─────────────────────────────────────────────────────────────────────────────
    TestCase(
        question="What is the hotel phone number?",
        expected_categories=["contact"],
        should_have_data=True,
        description="Phone number inquiry"
    ),
    TestCase(
        question="What is the hotel email address?",
        expected_categories=["contact"],
        should_have_data=True,
        description="Email inquiry"
    ),
    TestCase(
        question="Where is the hotel located?",
        expected_categories=["basic_info"],
        should_have_data=True,
        description="Hotel location"
    ),
    TestCase(
        question="What is the hotel address?",
        expected_categories=["basic_info"],
        should_have_data=True,
        description="Hotel address"
    ),
    
    # ─────────────────────────────────────────────────────────────────────────────
    # GERMAN LANGUAGE QUESTIONS
    # ─────────────────────────────────────────────────────────────────────────────
    TestCase(
        question="Wie viel kostet das Parken?",
        expected_categories=["parking"],
        should_have_data=True,
        description="German: Parking cost"
    ),
    TestCase(
        question="Gibt es ein Fitnessstudio?",
        expected_categories=["fitness_wellness"],
        should_have_data=True,
        description="German: Gym availability"
    ),
    TestCase(
        question="Wie viele Zimmer hat das Hotel?",
        expected_categories=["rooms"],
        should_have_data=True,
        description="German: Room count"
    ),
    
    # ─────────────────────────────────────────────────────────────────────────────
    # NEGATIVE TESTS (Should return no data)
    # ─────────────────────────────────────────────────────────────────────────────
    TestCase(
        question="What is the weather forecast?",
        expected_categories=[],
        should_have_data=False,
        description="Irrelevant: Weather"
    ),
    TestCase(
        question="What restaurants are nearby?",
        expected_categories=[],
        should_have_data=False,
        description="Irrelevant: External restaurants"
    ),
    TestCase(
        question="What is the capital of France?",
        expected_categories=[],
        should_have_data=False,
        description="Irrelevant: General knowledge"
    ),
    TestCase(
        question="How do I book a flight?",
        expected_categories=[],
        should_have_data=False,
        description="Irrelevant: Flight booking"
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Test Runner
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    test_case: TestCase
    passed: bool
    actual_categories: list[str]
    has_data: bool
    similarity_scores: list[float]
    error: Optional[str] = None


def run_test(test_case: TestCase) -> TestResult:
    """Run a single test case against the RAG API."""
    try:
        response = requests.post(
            API_URL,
            json={"question": test_case.question},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        actual_categories = data.get("categories", [])
        has_data = data.get("has_relevant_data", False)
        
        # Extract similarity scores
        scores = [
            item.get("similarity_score", 0) 
            for item in data.get("relevant_data", [])
        ]
        
        # Determine if test passed
        if test_case.should_have_data:
            # Check if at least one expected category is present
            category_match = any(
                cat in actual_categories 
                for cat in test_case.expected_categories
            )
            passed = has_data and category_match
        else:
            # For negative tests, we expect no relevant data
            passed = not has_data
        
        return TestResult(
            test_case=test_case,
            passed=passed,
            actual_categories=actual_categories,
            has_data=has_data,
            similarity_scores=scores
        )
        
    except Exception as e:
        return TestResult(
            test_case=test_case,
            passed=False,
            actual_categories=[],
            has_data=False,
            similarity_scores=[],
            error=str(e)
        )


def print_results(results: list[TestResult]):
    """Print test results in a nicely formatted way."""
    
    # Header
    print("\n")
    print("╔" + "═" * 98 + "╗")
    print("║" + " HOTEL RAG TEST RESULTS ".center(98) + "║")
    print("╚" + "═" * 98 + "╝")
    print()
    
    # Group results by category
    passed_tests = [r for r in results if r.passed]
    failed_tests = [r for r in results if not r.passed]
    
    # Summary
    total = len(results)
    passed = len(passed_tests)
    failed = len(failed_tests)
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    print("┌" + "─" * 40 + "┐")
    print("│" + " SUMMARY ".center(40) + "│")
    print("├" + "─" * 40 + "┤")
    print(f"│  Total Tests:    {total:>4}                  │")
    print(f"│  Passed:         {passed:>4}  ✓                │")
    print(f"│  Failed:         {failed:>4}  ✗                │")
    print(f"│  Pass Rate:      {pass_rate:>5.1f}%               │")
    print("└" + "─" * 40 + "┘")
    print()
    
    # Detailed Results Table
    print("┌" + "─" * 98 + "┐")
    print("│" + " DETAILED RESULTS ".center(98) + "│")
    print("├" + "─" * 4 + "┬" + "─" * 45 + "┬" + "─" * 20 + "┬" + "─" * 12 + "┬" + "─" * 12 + "┤")
    print(f"│ {'ST':<2} │ {'Question':<43} │ {'Categories':<18} │ {'Score':<10} │ {'Expected':<10} │")
    print("├" + "─" * 4 + "┼" + "─" * 45 + "┼" + "─" * 20 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┤")
    
    for result in results:
        status = "✓ " if result.passed else "✗ "
        question = result.test_case.question[:41] + ".." if len(result.test_case.question) > 43 else result.test_case.question
        
        if result.error:
            cats = "ERROR"
            score = "N/A"
        else:
            cats = ",".join(result.actual_categories[:2])[:18] if result.actual_categories else "—"
            score = f"{result.similarity_scores[0]:.2f}" if result.similarity_scores else "—"
        
        expected = ",".join(result.test_case.expected_categories[:2])[:10] if result.test_case.expected_categories else "none"
        
        print(f"│ {status:<2} │ {question:<43} │ {cats:<18} │ {score:<10} │ {expected:<10} │")
    
    print("└" + "─" * 4 + "┴" + "─" * 45 + "┴" + "─" * 20 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┘")
    print()
    
    # Failed Tests Details
    if failed_tests:
        print("┌" + "─" * 98 + "┐")
        print("│" + " FAILED TESTS DETAILS ".center(98) + "│")
        print("└" + "─" * 98 + "┘")
        print()
        
        for i, result in enumerate(failed_tests, 1):
            print(f"  {i}. {result.test_case.description}")
            print(f"     Question: \"{result.test_case.question}\"")
            print(f"     Expected: categories={result.test_case.expected_categories}, should_have_data={result.test_case.should_have_data}")
            print(f"     Actual:   categories={result.actual_categories}, has_data={result.has_data}")
            if result.error:
                print(f"     Error: {result.error}")
            print()
    
    # Category Performance
    print("┌" + "─" * 50 + "┐")
    print("│" + " PERFORMANCE BY CATEGORY ".center(50) + "│")
    print("├" + "─" * 20 + "┬" + "─" * 10 + "┬" + "─" * 10 + "┬" + "─" * 7 + "┤")
    print(f"│ {'Category':<18} │ {'Passed':<8} │ {'Failed':<8} │ {'Rate':<5} │")
    print("├" + "─" * 20 + "┼" + "─" * 10 + "┼" + "─" * 10 + "┼" + "─" * 7 + "┤")
    
    categories = {}
    for result in results:
        for cat in result.test_case.expected_categories or ["none"]:
            if cat not in categories:
                categories[cat] = {"passed": 0, "failed": 0}
            if result.passed:
                categories[cat]["passed"] += 1
            else:
                categories[cat]["failed"] += 1
    
    for cat, stats in sorted(categories.items()):
        total_cat = stats["passed"] + stats["failed"]
        rate = (stats["passed"] / total_cat * 100) if total_cat > 0 else 0
        print(f"│ {cat:<18} │ {stats['passed']:<8} │ {stats['failed']:<8} │ {rate:>4.0f}% │")
    
    print("└" + "─" * 20 + "┴" + "─" * 10 + "┴" + "─" * 10 + "┴" + "─" * 7 + "┘")
    print()
    
    # Score Distribution
    all_scores = [s for r in results for s in r.similarity_scores]
    if all_scores:
        print("┌" + "─" * 40 + "┐")
        print("│" + " SIMILARITY SCORE STATS ".center(40) + "│")
        print("├" + "─" * 40 + "┤")
        print(f"│  Min Score:      {min(all_scores):>6.3f}               │")
        print(f"│  Max Score:      {max(all_scores):>6.3f}               │")
        print(f"│  Avg Score:      {sum(all_scores)/len(all_scores):>6.3f}               │")
        print("└" + "─" * 40 + "┘")
    
    print()
    print("═" * 100)
    if failed == 0:
        print("  🎉 ALL TESTS PASSED!")
    else:
        print(f"  ⚠️  {failed} test(s) failed. Review the details above.")
    print("═" * 100)
    print()


def main():
    """Run all tests and display results."""
    print("\n🔍 Running RAG Test Suite...")
    print(f"   Testing {len(TEST_CASES)} questions against {API_URL}\n")
    
    results = []
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"   [{i:>2}/{len(TEST_CASES)}] Testing: {test_case.description}...", end=" ", flush=True)
        result = run_test(test_case)
        results.append(result)
        print("✓" if result.passed else "✗")
    
    print_results(results)


if __name__ == "__main__":
    main()

