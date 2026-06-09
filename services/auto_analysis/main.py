"""Main service for auto-analysis of annotated data with auto-save and resume."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    GEMINI_API_KEY,
    GUIDELINE_PATH,
    SCRAPED_DATA_PATH,
    OUTPUT_DIR,
    BATCH_SIZE,
)
from utils import load_guideline, load_scraped_data, chunk_records, save_yaml, load_yaml
from gemini_validator import GeminiValidator


class AutoAnalysisService:
    """Service for validating and correcting annotated data using Gemini API.
    
    Features:
    - Auto-saves results every batch
    - Can resume from where it left off if interrupted
    - Tracks processing state in progress.json
    """

    def __init__(self):
        """Initialize the service."""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        self.guideline = load_guideline(GUIDELINE_PATH)
        self.validator = GeminiValidator(GEMINI_API_KEY, self.guideline)
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking files
        self.progress_file = self.output_dir / "progress.json"
        self.incorrect_items_file = self.output_dir / "incorrect_items.yaml"
        self.summary_file = self.output_dir / "summary.yaml"

    def _load_progress(self) -> Dict:
        """Load processing progress if it exists."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "started_at": datetime.now().isoformat(),
            "last_batch": 0,
            "total_incorrect": 0,
            "processed_batches": 0,
            "failed_batches": 0,
        }

    def _save_progress(self, progress: Dict) -> None:
        """Save processing progress to JSON file."""
        progress["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def _load_incorrect_items(self) -> List[Dict]:
        """Load any previously saved incorrect items."""
        if self.incorrect_items_file.exists():
            data = load_yaml(self.incorrect_items_file)
            return data.get("records", [])
        return []

    def _save_incorrect_items(self, incorrect_items: List[Dict]) -> None:
        """Save incorrect items in YAML format."""
        data = {"records": incorrect_items}
        save_yaml(data, self.incorrect_items_file)

    def _save_summary(self, progress: Dict, total_items: int) -> None:
        """Save processing summary."""
        total_incorrect = progress["total_incorrect"]
        processed = progress["processed_batches"]
        total_batches = progress.get("total_batches", 1052)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "started_at": progress.get("started_at"),
            "total_records": total_items,
            "total_incorrect": total_incorrect,
            "accuracy": f"{((total_items - total_incorrect) / total_items * 100):.2f}%" if total_items > 0 else "0%",
            "batches_processed": processed,
            "batches_failed": progress["failed_batches"],
            "total_batches": total_batches,
            "is_complete": processed >= total_batches,
            "status": "COMPLETE" if processed >= total_batches else "IN_PROGRESS",
        }

        save_yaml(summary, self.summary_file)

    async def process_all_data(self, resume: bool = True) -> None:
        """Process all scraped data in batches with auto-save and resume capability.
        
        Args:
            resume: If True, resume from where it left off. If False, start fresh.
        """
        print(f"Loading scraped data from {SCRAPED_DATA_PATH}...")
        records = load_scraped_data(SCRAPED_DATA_PATH)
        print(f"Loaded {len(records)} records")

        # Split into batches
        batches = chunk_records(records, BATCH_SIZE)
        print(f"Split into {len(batches)} batches of {BATCH_SIZE} items")

        # Load progress if resuming
        progress = self._load_progress()
        start_batch = 0
        all_incorrect = self._load_incorrect_items()

        if resume and progress["last_batch"] > 0:
            start_batch = progress["last_batch"]
            print(f"\n🔄 Resuming from batch {start_batch}...")
            print(f"   Previously found incorrect items: {progress['total_incorrect']}")
            print(f"   Already processed: {progress['processed_batches']} batches\n")
        else:
            print(f"\n▶️  Starting fresh processing...\n")
            progress["started_at"] = datetime.now().isoformat()
            all_incorrect = []

        progress["total_batches"] = len(batches)

        # Process batches with concurrency control
        print(f"Processing batches {start_batch + 1} to {len(batches)}...\n")

        try:
            # Process batches in chunks of 10 to maintain concurrency limit
            for chunk_idx in range(start_batch, len(batches), 10):
                batch_chunk = batches[chunk_idx : chunk_idx + 10]
                batch_numbers = range(chunk_idx + 1, chunk_idx + len(batch_chunk) + 1)

                # Run concurrent validation for this chunk
                tasks = [
                    self.validator.validate_batch(batch, batch_num)
                    for batch, batch_num in zip(batch_chunk, batch_numbers)
                ]

                results = await asyncio.gather(*tasks)

                # Process results and auto-save after each chunk
                for result in results:
                    batch_num = result["batch_number"]
                    if result["status"] == "success":
                        incorrect_count = result.get("incorrect_count", 0)
                        progress["total_incorrect"] += incorrect_count
                        progress["processed_batches"] += 1
                        progress["last_batch"] = batch_num

                        if incorrect_count > 0:
                            new_items = result["incorrect_items"]
                            all_incorrect.extend(new_items)
                            
                            # Auto-save after each batch with new items
                            self._save_incorrect_items(all_incorrect)

                        print(
                            f"✓ Batch {batch_num:5d}: {result['total_items']:3d} items, "
                            f"{incorrect_count:3d} incorrect | "
                            f"Total: {progress['total_incorrect']} incorrect"
                        )
                    else:
                        progress["failed_batches"] += 1
                        progress["last_batch"] = batch_num
                        print(
                            f"✗ Batch {batch_num:5d}: {result['status']} - {result.get('error', '')}"
                        )

                    # Save progress after each batch (important for recovery)
                    self._save_progress(progress)
                    
                    # Also save summary regularly
                    self._save_summary(progress, len(records))

            # Final save
            print(f"\n✅ Processing complete!")
            print(f"Total batches processed: {progress['processed_batches']}")
            print(f"Total incorrect items: {progress['total_incorrect']}")
            print(f"Failed batches: {progress['failed_batches']}")

            if all_incorrect:
                self._save_incorrect_items(all_incorrect)
                print(f"✓ Saved {len(all_incorrect)} incorrect items to: {self.incorrect_items_file}")
            else:
                print("✓ All items are correctly labeled!")

            # Final summary
            self._save_summary(progress, len(records))
            print(f"✓ Summary saved to: {self.summary_file}")
            
            # Show progress file location
            print(f"✓ Progress tracking: {self.progress_file}")

        except KeyboardInterrupt:
            print("\n\n⚠️  Processing interrupted by user!")
            self._save_progress(progress)
            self._save_incorrect_items(all_incorrect)
            self._save_summary(progress, len(records))
            print(f"\n📊 Progress saved:")
            print(f"   Processed: {progress['processed_batches']} batches")
            print(f"   Failed: {progress['failed_batches']} batches")
            print(f"   Incorrect items found: {progress['total_incorrect']}")
            print(f"\n🔄 To resume: Run the script again and it will continue from batch {progress['last_batch']}")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            self._save_progress(progress)
            self._save_incorrect_items(all_incorrect)
            self._save_summary(progress, len(records))
            print(f"\n💾 Progress saved before error:")
            print(f"   Processed: {progress['processed_batches']} batches")
            print(f"   Incorrect items: {progress['total_incorrect']}")
            print(f"\n🔄 To resume: Run the script again")
            raise


async def main():
    """Main entry point."""
    try:
        service = AutoAnalysisService()
        
        # Check if there's existing progress
        progress = service._load_progress()
        if progress["last_batch"] > 0:
            print("Found existing progress file.")
            print(f"Last processed batch: {progress['last_batch']}")
            print(f"Incorrect items found so far: {progress['total_incorrect']}\n")
            
            # Ask user if they want to resume
            resume = input("Resume from last batch? (y/n, default=y): ").lower() != 'n'
        else:
            resume = False
            
        await service.process_all_data(resume=resume)
    except KeyboardInterrupt:
        print("\n\nShutdown requested by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
