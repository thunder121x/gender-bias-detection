from datetime import datetime
from .constants import DATETIME_FORMAT

# Define the CSV output fields
FIELDS = [
    "id", "platform", "platform_type", "url", "content_type",
    "timestamp", "scraper_module", "text", "raw_text", "scrape_date"
]
SCRAPE_DATE = datetime.utcnow().strftime(DATETIME_FORMAT)