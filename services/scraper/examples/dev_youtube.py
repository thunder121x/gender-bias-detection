
import asyncio
import os
from scraper.vdo_based import YouTubeScraper, TikTokScraper

# if __name__ == "__main__":
#     video_id = 7546438874688523538
#     ms_token = os.environ.get("ms_token", None)

#     scraper = TikTokScraper(video_id=video_id, ms_token=ms_token)
#     asyncio.run(scraper.run())


if __name__ == "__main__":
    video_id = "kvQbz6ZXGi0"
    scraper = YouTubeScraper(video_id)
    scraper.run()