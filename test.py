import os
import csv
import asyncio
from datetime import datetime
from pathlib import Path
import time
from typing import List
import uuid
from TikTokApi import TikTokApi

from scraper.constants import DATETIME_FORMAT, OUTPUT_DIR_TIKTOK
from scraper.dataclass import SocialMediaRecord


class TikTokExtractor:
    """Encapsulates TikTok extraction methods for videos, users, trends, and hashtags."""

    def __init__(self, ms_token: str | None = None):
        self.ms_token = ms_token or os.environ.get("ms_token")
        self.scraper_module = "TikTokApi"
        self.scrape_date = datetime.utcnow().strftime(DATETIME_FORMAT)
        self.output_dir = OUTPUT_DIR_TIKTOK
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # 🧩 Shared helper: collect comments from async list of videos
    # ----------------------------------------------------------------------
    async def _collect_comments_from_videos(
        self,
        videos_async_iter,
        platform_type: str,
        base_url: str,
        tag_or_user: str | None = None,
        comments_count: int = 10,
    ) -> List[SocialMediaRecord]:
        """Collect comments from a stream of videos asynchronously."""
        comments: List[SocialMediaRecord] = []
        async for v in videos_async_iter:
            async for c in v.comments(count=comments_count):
                data = c.as_dict
                if platform_type == "hashtag":
                    url = f"{base_url}/{tag_or_user}/video/{v.id}"
                elif platform_type == "user":
                    url = f"{base_url}/{tag_or_user}/video/{v.id}"
                else:
                    url = f"{base_url}/{v.id}"

                comments.append(
                    SocialMediaRecord(
                        id=data.get("cid"),
                        platform="tiktok",
                        platform_type=platform_type,
                        url=url,
                        content_type="comment",
                        timestamp=data.get("create_time"),
                        scraper_module=self.scraper_module,
                        text=data.get("text", "").strip(),
                        raw_text=data.get("text", ""),
                        scrape_date=self.scrape_date,
                    )
                )
        return comments

    # ----------------------------------------------------------------------
    async def _save_to_csv(self, records: List[SocialMediaRecord], filename: str):
        """Save extracted data to CSV."""
        if not records:
            print("⚠️ No records to save.")
            return
        output_file = self.output_dir / filename
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SocialMediaRecord.get_fields())
            writer.writeheader()
            for r in records:
                writer.writerow(r.to_dict())
        print(f"✅ Saved {len(records)} records to {output_file}")

    # ----------------------------------------------------------------------
    # 1️⃣ Extract comments from a specific video
    # ----------------------------------------------------------------------
    async def video_id(self, video_id: int, comments_count: int = 30):
        async with TikTokApi() as api:
            await api.create_sessions(
                num_sessions=1,
                ms_tokens=[self.ms_token],
                sleep_after=3,
                browser="chromium",
                headless=False,
            )
            video = api.video(id=video_id)
            video_url = f"https://www.tiktok.com/video/{video_id}"
            comments = []
            async for c in video.comments(count=comments_count):
                data = c.as_dict
                comments.append(
                    SocialMediaRecord(
                        id=data.get("cid"),
                        platform="tiktok",
                        platform_type="video",
                        url=video_url,
                        content_type="comment",
                        timestamp=data.get("create_time"),
                        scraper_module=self.scraper_module,
                        text=data.get("text", "").strip(),
                        raw_text=data.get("text", ""),
                        scrape_date=self.scrape_date,
                    )
                )

        await self._save_to_csv(comments, f"tiktok_comments_{video_id}.csv")

    # ----------------------------------------------------------------------
    # 2️⃣ Extract comments from user videos
    # ----------------------------------------------------------------------
    async def user(self, user: str, videos_count: int = 5, comments_count: int = 10):
        async with TikTokApi() as api:
            await api.create_sessions(
                num_sessions=1,
                ms_tokens=[self.ms_token],
                sleep_after=3,
                browser="chromium",
                headless=False,
                starting_url=f"https://www.tiktok.com/@{user}"
            )
            u = api.user(user)
            videos_iter = u.videos(count=videos_count)
            comments = await self._collect_comments_from_videos(
                videos_iter,
                platform_type="user",
                base_url="https://www.tiktok.com/@",
                tag_or_user=user,
                comments_count=comments_count,
            )
        await self._save_to_csv(comments, f"tiktok_user_{user}_comments.csv")

    # ----------------------------------------------------------------------
    # 3️⃣ Extract trending videos and comments
    # ----------------------------------------------------------------------
    async def trend(self, videos_count: int = 5, comments_count: int = 10):
        async with TikTokApi() as api:
            await api.create_sessions(
                num_sessions=1,
                ms_tokens=[self.ms_token],
                sleep_after=3,
                browser="chromium",
                headless=False,
            )
            videos_iter = api.trending.videos(count=videos_count)
            comments = await self._collect_comments_from_videos(
                videos_iter,
                platform_type="trend",
                base_url="https://www.tiktok.com/video",
                comments_count=comments_count,
            )
        # Generate unique filename
        uid = uuid.uuid4().hex[:8]  # short unique id
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"tiktok_trending_comments_{uid}_{timestamp}.csv"
        await self._save_to_csv(comments, filename)

    # ----------------------------------------------------------------------
    # 4️⃣ Extract hashtag videos and comments
    # ----------------------------------------------------------------------
    async def hashtag(self, tag: str, videos_count: int = 5, comments_count: int = 10):
        async with TikTokApi() as api:
            await api.create_sessions(
                num_sessions=1,
                ms_tokens=[self.ms_token],
                sleep_after=10,
                browser="chromium",
                headless=False,
                starting_url=f"https://www.tiktok.com/tag/{tag}"
            )
            hashtag = api.hashtag(name=tag)
            videos_iter = hashtag.videos(count=videos_count)
            comments = await self._collect_comments_from_videos(
                videos_iter,
                platform_type="hashtag",
                base_url="https://www.tiktok.com/tag",
                tag_or_user=tag,
                comments_count=comments_count,
            )
        await self._save_to_csv(comments, f"tiktok_hashtag_{tag}.csv")

    # ----------------------------------------------------------------------
    # 5. Extract hashtag videos and comments
    # ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# ✅ Usage Example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    async def main():
        extractor = TikTokExtractor()
        # await extractor.hashtag(tag="fyp", videos_count=20, comments_count=20)
        # await extractor.user(user="ckfastwork", videos_count=5, comments_count=50)
        await extractor.trend(videos_count=15, comments_count=50)

    asyncio.run(main())