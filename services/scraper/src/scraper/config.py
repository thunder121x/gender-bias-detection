from datetime import datetime
from typing import Literal, TypeAlias
from .constants import DATETIME_FORMAT

# Define the CSV output fields
FIELDS: list[str] = [
    "id", "platform", "platform_type", "url", "content_type",
    "timestamp", "scraper_module", "text", "raw_text", "scrape_date"
]
SUPPORTED_PLATFORMS: tuple[str, ...] = ("tiktok", "youtube", "twitter", "facebook")

PlatformType: TypeAlias = Literal["tiktok", "youtube", "twitter", "facebook"]

SCRAPE_DATE = datetime.utcnow().strftime(DATETIME_FORMAT)