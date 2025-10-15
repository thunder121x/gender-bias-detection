import os
import csv
import asyncio
from datetime import datetime
from pathlib import Path
from TikTokApi import TikTokApi
from typing import List

from scraper.constants import DATETIME_FORMAT, OUTPUT_DIR_TIKTOK
from scraper.utils import date_time_formatter
from scraper.dataclass import SocialMediaRecord


class TikTokScraper:
    """Scraper class for extracting TikTok comments using TikTokApi."""

    def __init__(self, video_id: int, ms_token: str | None = None):
        self.video_id = video_id
        self.ms_token = ms_token or os.environ.get("ms_token", None)
        self.scraper_module = "TikTokApi"
        self.scrape_date = datetime.utcnow().strftime(DATETIME_FORMAT)
        self.output_dir = OUTPUT_DIR_TIKTOK
        self.output_file = self.output_dir / f"tiktok_comments_{video_id}.csv"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def extract_comments(self, count: int = 30) -> List[SocialMediaRecord]:
        """Asynchronously extract comments from a TikTok video."""
        comments: List[SocialMediaRecord] = []

        async with TikTokApi() as api:
            await api.create_sessions(
                num_sessions=1,
                ms_tokens=[self.ms_token],
                sleep_after=3,
                browser="chromium",
                headless=False,
                override_browser_args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                    "--window-size=1280,800"
                ],
                starting_url="https://www.tiktok.com/@tiktok"
            )

            video = api.video(id=self.video_id)
            video_url = f"https://www.tiktok.com/@tiktok/video/{self.video_id}"

            async for c in video.comments(count=count):
                data = c.as_dict
                comment_id = data.get("cid")
                text = data.get("text", "").strip()
                timestamp = data.get("create_time")

                record = SocialMediaRecord(
                    id=comment_id,
                    platform="tiktok",
                    platform_type="video",
                    url=video_url,
                    content_type="comment",
                    timestamp=timestamp,
                    scraper_module=self.scraper_module,
                    text=text,
                    raw_text=data.get("text", ""),
                    scrape_date=self.scrape_date,
                )
                comments.append(record)

        return comments

    async def save_to_csv(self, comments: List[SocialMediaRecord]):
        """Write extracted TikTok comments to a CSV file."""
        with open(self.output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SocialMediaRecord.get_fields())
            writer.writeheader()
            for record in comments:
                writer.writerow(record.to_dict())
        print(f"✅ Saved {len(comments)} comments to {self.output_file}")

    async def run(self):
        """Run the full TikTok scraping pipeline."""
        comments = await self.extract_comments()
        if comments:
            await self.save_to_csv(comments)
        else:
            print("⚠️ No comments found for this video.")
            