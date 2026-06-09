"""Example/demo script showing the auto-analysis service in action."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import GEMINI_API_KEY, GUIDELINE_PATH
from utils import load_guideline, load_scraped_data, chunk_records
from gemini_validator import GeminiValidator


async def demo_batch_validation():
    """Run a demo with just the first 2 batches to show how it works."""

    print("=" * 70)
    print("AUTO-ANALYSIS SERVICE - DEMO")
    print("=" * 70)
    print("\nThis demo shows how the service validates annotations.\n")

    # Load data
    from config import SCRAPED_DATA_PATH, BATCH_SIZE

    guideline = load_guideline(GUIDELINE_PATH)
    records = load_scraped_data(SCRAPED_DATA_PATH)
    batches = chunk_records(records, BATCH_SIZE)

    print(f"Total records in dataset: {len(records)}")
    print(f"Running demo with first 2 batches of {BATCH_SIZE} items each\n")

    # Initialize validator
    validator = GeminiValidator(GEMINI_API_KEY, guideline)

    # Process first 2 batches
    for batch_num in range(1, 3):
        batch = batches[batch_num - 1]
        print(f"Processing Batch {batch_num}...")
        print(f"  Items: {len(batch)}")
        print(f"  Sample items:")

        for i, item in enumerate(batch[:3]):
            print(f"    {i+1}. ID: {item['id'][:20]}...")
            print(f"       Text: {item['text'][:50]}...")
            print(f"       Label: {item['predicted_label']}")

        # Validate batch
        result = await validator.validate_batch(batch, batch_num)

        print(f"\n  Validation Result:")
        print(f"    Status: {result['status']}")
        print(f"    Total items in batch: {result.get('total_items', 'N/A')}")
        print(f"    Incorrect items: {result.get('incorrect_count', 0)}")

        if result.get("incorrect_items"):
            print(f"    First incorrect item:")
            item = result["incorrect_items"][0]
            print(f"      ID: {item['id']}")
            print(f"      Text: {item['text'][:60]}...")
            print(f"      Predicted: {item['predicted_label']}")
            print(f"      Correct: {item['correct_label']}")
            print(f"      Reason: {item['reason'][:80]}...")

        print()

    print("=" * 70)
    print("Demo complete! To run full processing: python main.py")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_batch_validation())
