from TikTokApi import TikTokApi
import asyncio
import os

video_id = 7546438874688523538
ms_token = os.environ.get("ms_token", None)  # set your own ms_token


async def get_comments():
    async with TikTokApi() as api:
        await api.create_sessions(
            num_sessions=1,
            ms_tokens=[ms_token],
            sleep_after=3,
            browser="chromium",
            headless=False,  # 👈 must be visible
            override_browser_args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--window-size=1280,800"
            ],
            starting_url="https://www.tiktok.com/@tiktok"
        )
        video = api.video(id=video_id)
        count = 0
        async for comment in video.comments(count=30):
            print(comment)
            print(comment.as_dict)


if __name__ == "__main__":
    asyncio.run(get_comments())
