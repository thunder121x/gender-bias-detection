from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal, Optional, ClassVar
from .config import FIELDS, SCRAPE_DATE
from .utils import date_time_formatter
from .constants import DATETIME_FORMAT


@dataclass
class SocialMediaRecord:
    """
    Represents a single scraped social media item used in the
    gender bias detection dataset.

    FIELDS:
        id              – unique identifier of the record
        platform        – name of the platform (e.g., 'tiktok', 'twitter', 'facebook')
        platform_type   – type of content source (e.g., 'comment', 'post', 'reply')
        url             – URL to the original content
        content_type    – 'text', 'video', or 'image'
        timestamp       – original post timestamp (ISO 8601 format)
        scraper_module  – scraper name or source module
        text            – cleaned text for NLP processing
        raw_text        – original raw text before preprocessing
        scrape_date     – datetime when the data was collected
    """

    id: Optional[str] = None
    platform: Optional[str] = None
    platform_type: Optional[str] = None
    url: Optional[str] = None
    content_type: Optional[str] = "text"
    timestamp: Optional[str] = None
    scraper_module: Optional[str] = None
    text: Optional[str] = None
    raw_text: Optional[str] = None
    scrape_date: str = field(default_factory=lambda: SCRAPE_DATE)

    # class-level constants
    FIELDS: ClassVar[list[str]] = FIELDS
    DATETIME_FORMAT: ClassVar[str] = DATETIME_FORMAT

    def __init__(
        self,
        id: Optional[str] = None,
        platform: Optional[Literal["tiktok", "youtube", "twitter", "facebook"]] = None,
        platform_type: Optional[str] = None,
        url: Optional[str] = None,
        content_type: Optional[str] = "text",
        timestamp: Optional[int | str] = None,
        scraper_module: Optional[str] = None,
        text: Optional[str] = None,
        raw_text: Optional[str] = None,
        scrape_date: Optional[str] = None,
    ):
        """Initialize the record with standardized timestamp handling."""
        if isinstance(timestamp, int):
            timestamp_str = date_time_formatter(timestamp)
        elif isinstance(timestamp, str):
            timestamp_str = timestamp
        else:
            timestamp_str = ""

        self.id = id
        self.platform = platform
        self.platform_type = platform_type
        self.url = url
        self.content_type = content_type
        self.timestamp = timestamp_str
        self.scraper_module = scraper_module
        self.text = text
        self.raw_text = raw_text
        self.scrape_date = scrape_date or SCRAPE_DATE

    def to_dict(self) -> dict:
        """Convert dataclass to a dictionary matching CSV output."""
        return asdict(self)

    @classmethod
    def get_fields(cls) -> list[str]:
        """Return standardized CSV header fields."""
        return cls.FIELDS