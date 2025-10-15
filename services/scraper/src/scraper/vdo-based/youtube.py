import csv
from datetime import datetime
from yt_dlp import YoutubeDL
from pathlib import Path
from typing import List

from dataclasses import dataclass, asdict
from scraper.constants import DATETIME_FORMAT, OUTPUT_DIR_YOUTUBE
from scraper.config import FIELDS
from scraper.utils import date_time_formatter
from scraper.dataclass import SocialMediaRecord


@dataclass
class YouTubeScraper:
    """Scraper class for extracting YouTube comments using yt-dlp."""

    video_url: str
    output_dir: Path = OUTPUT_DIR_YOUTUBE
    scraper_module: str = "yt-dlp"

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scrape_date = datetime.utcnow().strftime(DATETIME_FORMAT)
        self.output_file = self.output_dir / "youtube_comments.csv"

    def get_options(self) -> dict:
        """Return standard yt-dlp extraction options."""
        return {
            'skip_download': True,
            'extract_flat': False,
            'quiet': True,
            'force_generic_extractor': False,
            'extractor_args': {'youtube': {'skip': ['dash']}},
            'noplaylist': True,
            'writeinfojson': True,
            'getcomments': True,
            'dump_single_json': True
        }

    def extract_comments(self) -> List[SocialMediaRecord]:
        """Extract comments from the YouTube video."""
        comments: List[SocialMediaRecord] = []

        with YoutubeDL(self.get_options()) as ydl:
            result = ydl.extract_info(self.video_url, download=False)
            raw_comments = result.get("comments", [])

        for c in raw_comments:
            record = SocialMediaRecord(
                id=c.get("id"),
                platform="youtube",
                platform_type="video",
                url=self.video_url,
                content_type="comment",
                timestamp=c.get("timestamp"),
                scraper_module=self.scraper_module,
                text=c.get("text", "").strip(),
                raw_text=c.get("text", ""),
                scrape_date=self.scrape_date
            )
            comments.append(record)

        return comments

    def save_to_csv(self, comments: List[SocialMediaRecord]):
        """Write extracted comments to CSV file."""
        with open(self.output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SocialMediaRecord.get_fields())
            writer.writeheader()
            for record in comments:
                writer.writerow(record.to_dict())
        print(f"✅ Saved {len(comments)} comments to {self.output_file}")

    def run(self):
        """Run the full YouTube comment scraping pipeline."""
        comments = self.extract_comments()
        if comments:
            self.save_to_csv(comments)
        else:
            print("⚠️ No comments found for this video.")


if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=bQoVZe4Ohbo"
    scraper = YouTubeScraper(video_url)
    scraper.run()