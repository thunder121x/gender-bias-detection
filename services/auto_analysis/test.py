"""Test script for auto-analysis service."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    GEMINI_API_KEY,
    GUIDELINE_PATH,
    SCRAPED_DATA_PATH,
    BATCH_SIZE,
)
from utils import load_guideline, load_scraped_data, chunk_records


def test_configuration():
    """Test if configuration is valid."""
    print("Testing configuration...")

    # Check API key
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not set in environment")
        return False
    print("✓ GEMINI_API_KEY is configured")

    # Check guideline file
    if not GUIDELINE_PATH.exists():
        print(f"❌ Guideline file not found: {GUIDELINE_PATH}")
        return False
    print(f"✓ Guideline file exists: {GUIDELINE_PATH}")

    # Check scraped data file
    if not SCRAPED_DATA_PATH.exists():
        print(f"❌ Scraped data file not found: {SCRAPED_DATA_PATH}")
        return False
    print(f"✓ Scraped data file exists: {SCRAPED_DATA_PATH}")

    return True


def test_data_loading():
    """Test if data can be loaded correctly."""
    print("\nTesting data loading...")

    try:
        guideline = load_guideline(GUIDELINE_PATH)
        guideline_length = len(guideline)
        print(f"✓ Guideline loaded: {guideline_length} characters")

        records = load_scraped_data(SCRAPED_DATA_PATH)
        print(f"✓ Loaded {len(records)} records")

        # Validate record structure
        if records:
            sample = records[0]
            required_keys = {"id", "text", "predicted_label"}
            if all(key in sample for key in required_keys):
                print(f"✓ Record structure is valid: {list(sample.keys())}")
            else:
                print(f"❌ Record missing required keys: {required_keys - set(sample.keys())}")
                return False

        # Test batching
        batches = chunk_records(records, BATCH_SIZE)
        print(f"✓ Records split into {len(batches)} batches of {BATCH_SIZE}")

        return True

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


def test_gemini_api():
    """Test if Gemini API can be accessed."""
    print("\nTesting Gemini API...")

    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")

        # Simple test
        response = model.generate_content("Say 'Ready for auto-analysis' in exactly 4 words.")
        print(f"✓ Gemini API responds: {response.text[:50]}...")

        return True

    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("AUTO-ANALYSIS SERVICE - CONFIGURATION TEST")
    print("=" * 60)

    results = {
        "Configuration": test_configuration(),
        "Data Loading": test_data_loading(),
        "Gemini API": test_gemini_api(),
    }

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    print("=" * 60)

    if all_passed:
        print("✓ All tests passed! Ready to run: python main.py")
        return 0
    else:
        print("❌ Some tests failed. Fix issues above before running main.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
