#!/usr/bin/env python
# coding: utf-8

# In[45]:


from selenium import webdriver

driver = webdriver.Chrome()
driver.get("https://x.com/login")
print("Log in, then run the next cell to scrape a specific URL.")


# In[47]:


from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time, re
from urllib.parse import urljoin

def scrape_tweets(driver, target_url, max_scrolls=60, pause_secs=1.2):
    # 1) Always navigate to the intended page first (keeps your URL exactly as given)
    driver.get(target_url)

    # 2) Wait a bit for tweets (don’t hang forever)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'article[role="article"]'))
        )
    except TimeoutException:
        # Some pages (e.g., profiles with few posts) may need a scroll to kick in
        pass

    def get_article_elements(drv):
        return drv.find_elements(By.CSS_SELECTOR, 'article[role="article"]')

    def _num_from_label(s):
        if not s: return None
        s = s.strip().replace(",", "")
        m = re.search(r'([\d]+(?:\.\d+)?)\s*([KkMm])?', s)
        if not m: return None
        val = float(m.group(1))
        suf = (m.group(2) or "").lower()
        if suf == "k": val *= 1_000
        elif suf == "m": val *= 1_000_000
        return int(val)

    def _clean(t):
        return re.sub(r'\s+', ' ', t or '').strip()

    def parse_article(el):
        row = {
            "tweet_id": None, "timestamp_iso": None,
            "user_display_name": "", "user_handle": "",
            "text": "", "like_count": None, "retweet_count": None,
            "reply_count": None, "view_count": None,
            "permalink": "", "key": None
        }

        # --- user display name & handle (robust across UI variants) ---
        try:
            # Try both legacy and new testids
            blocks = el.find_elements(By.CSS_SELECTOR, 'div[data-testid="User-Name"], div[data-testid="User-Names"]')
            spans = []
            if blocks:
                spans = blocks[0].find_elements(By.CSS_SELECTOR, "span")

            # Extract handle and display name from spans if available
            handle_found = False
            for s in spans:
                txt = _clean(s.text)
                if not txt:
                    continue
                if txt.startswith("@"):
                    row["user_handle"] = txt
                    handle_found = True
                elif not row["user_display_name"]:
                    row["user_display_name"] = txt

            # Fallback: search any span with @ in the whole article
            if not handle_found:
                try:
                    handle_span = el.find_element(By.XPATH, './/span[starts-with(normalize-space(text()), "@")]')
                    row["user_handle"] = _clean(handle_span.text)
                except NoSuchElementException:
                    pass

            # If name still empty, try the accessible name on the user link
            if not row["user_display_name"]:
                try:
                    # user link usually the first link to profile inside the header names block
                    user_links = el.find_elements(By.XPATH, './/a[starts-with(@href, "/")][@role="link"]')
                    if user_links:
                        aria = user_links[0].get_attribute("aria-label") or ""
                        # aria-label often like "Bonus (@sinsorn_bn)"
                        m = re.search(r'^(.*?)\s*\(@', aria)
                        if m:
                            row["user_display_name"] = _clean(m.group(1))
                except Exception:
                    pass
        except Exception:
            pass

        # text
        try:
            row["text"] = _clean(el.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]').text)
        except NoSuchElementException:
            # Some tweets are media-only; leave as empty
            pass

        # time
        try:
            row["timestamp_iso"] = el.find_element(By.CSS_SELECTOR, "time").get_attribute("datetime")
        except NoSuchElementException:
            pass

        # link & id
        try:
            a = el.find_elements(By.CSS_SELECTOR, 'a[href*="/status/"]')
            if a:
                href = a[0].get_attribute("href")
                if href.startswith("/"):
                    href = urljoin("https://x.com", href)
                row["permalink"] = href
                m = re.search(r"/status/(\d+)", href)
                if m:
                    row["tweet_id"] = m.group(1)
        except NoSuchElementException:
            pass

        # counts
        def c(testid):
            try:
                btn = el.find_element(By.CSS_SELECTOR, f'[data-testid="{testid}"]')
                lab = btn.get_attribute("aria-label") or btn.text
                return _num_from_label(lab)
            except NoSuchElementException:
                return None
        row["reply_count"]   = c("reply")
        row["retweet_count"] = c("retweet")
        row["like_count"]    = c("like")

        # views (best effort)
        try:
            for s in el.find_elements(By.CSS_SELECTOR, "span[aria-label]"):
                lab = s.get_attribute("aria-label") or ""
                if "View" in lab:
                    vc = _num_from_label(lab)
                    if vc is not None:
                        row["view_count"] = vc
                        break
        except Exception:
            pass

        row["key"] = row["tweet_id"] or f"{row['user_handle']}|{row['timestamp_iso']}|{row['text'][:50]}"
        return row

    results, seen = [], set()
    not_growing, last_seen = 0, 0

    for _ in range(max_scrolls):
        articles = get_article_elements(driver)
        for art in articles:
            row = parse_article(art)
            if not (row["tweet_id"] or row["text"]):
                continue
            if row["key"] in seen:
                continue
            seen.add(row["key"])
            results.append(row)
            print(
                f"[{row['timestamp_iso']}] {row['user_display_name']} {row['user_handle']} "
                f"♥{row['like_count']} ↻{row['retweet_count']} 💬{row['reply_count']} 👁️{row['view_count']}\n"
                f"{row['text']}\n{row['permalink']}\n" + "—"*40
            )

        if len(seen) == last_seen:
            not_growing += 1
        else:
            not_growing = 0
        last_seen = len(seen)
        if not_growing >= 6:
            break

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_secs)

    print(f"\nCollected {len(results)} unique tweets.")
    return results

# >>> Use it like this (keep your URL exactly as you wrote it):
TARGET_URL = "https://x.com/search?q=มันฮวา&f=live"  # or ANY Twitter/X link
results = scrape_tweets(driver, TARGET_URL, max_scrolls=80, pause_secs=1.2)


# In[ ]:


from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time, re
from urllib.parse import urljoin

def scrape_tweets(driver, target_url, max_scrolls=60, pause_secs=1.2):
    # 1) Always navigate to the intended page first (keeps your URL exactly as given)
    driver.get(target_url)

    # 2) Wait a bit for tweets (don’t hang forever)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'article[role="article"]'))
        )
    except TimeoutException:
        # Some pages (e.g., profiles with few posts) may need a scroll to kick in
        pass

    def get_article_elements(drv):
        return drv.find_elements(By.CSS_SELECTOR, 'article[role="article"]')

    def _num_from_label(s):
        if not s: return None
        s = s.strip().replace(",", "")
        m = re.search(r'([\d]+(?:\.\d+)?)\s*([KkMm])?', s)
        if not m: return None
        val = float(m.group(1))
        suf = (m.group(2) or "").lower()
        if suf == "k": val *= 1_000
        elif suf == "m": val *= 1_000_000
        return int(val)

    def _clean(t):
        return re.sub(r'\s+', ' ', t or '').strip()

    def parse_article(el):
        row = {
            "tweet_id": None, "timestamp_iso": None,
            "user_display_name": "", "user_handle": "",
            "text": "", "like_count": None, "retweet_count": None,
            "reply_count": None, "view_count": None,
            "permalink": "", "key": None
        }

        # --- user display name & handle (robust across UI variants) ---
        try:
            # Try both legacy and new testids
            blocks = el.find_elements(By.CSS_SELECTOR, 'div[data-testid="User-Name"], div[data-testid="User-Names"]')
            spans = []
            if blocks:
                spans = blocks[0].find_elements(By.CSS_SELECTOR, "span")

            # Extract handle and display name from spans if available
            handle_found = False
            for s in spans:
                txt = _clean(s.text)
                if not txt:
                    continue
                if txt.startswith("@"):
                    row["user_handle"] = txt
                    handle_found = True
                elif not row["user_display_name"]:
                    row["user_display_name"] = txt

            # Fallback: search any span with @ in the whole article
            if not handle_found:
                try:
                    handle_span = el.find_element(By.XPATH, './/span[starts-with(normalize-space(text()), "@")]')
                    row["user_handle"] = _clean(handle_span.text)
                except NoSuchElementException:
                    pass

            # If name still empty, try the accessible name on the user link
            if not row["user_display_name"]:
                try:
                    # user link usually the first link to profile inside the header names block
                    user_links = el.find_elements(By.XPATH, './/a[starts-with(@href, "/")][@role="link"]')
                    if user_links:
                        aria = user_links[0].get_attribute("aria-label") or ""
                        # aria-label often like "Bonus (@sinsorn_bn)"
                        m = re.search(r'^(.*?)\s*\(@', aria)
                        if m:
                            row["user_display_name"] = _clean(m.group(1))
                except Exception:
                    pass
        except Exception:
            pass

        # text
        try:
            row["text"] = _clean(el.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]').text)
        except NoSuchElementException:
            # Some tweets are media-only; leave as empty
            pass

        # time
        try:
            row["timestamp_iso"] = el.find_element(By.CSS_SELECTOR, "time").get_attribute("datetime")
        except NoSuchElementException:
            pass

        # link & id
        try:
            a = el.find_elements(By.CSS_SELECTOR, 'a[href*="/status/"]')
            if a:
                href = a[0].get_attribute("href")
                if href.startswith("/"):
                    href = urljoin("https://x.com", href)
                row["permalink"] = href
                m = re.search(r"/status/(\d+)", href)
                if m:
                    row["tweet_id"] = m.group(1)
        except NoSuchElementException:
            pass

        # counts
        def c(testid):
            try:
                btn = el.find_element(By.CSS_SELECTOR, f'[data-testid="{testid}"]')
                lab = btn.get_attribute("aria-label") or btn.text
                return _num_from_label(lab)
            except NoSuchElementException:
                return None
        row["reply_count"]   = c("reply")
        row["retweet_count"] = c("retweet")
        row["like_count"]    = c("like")

        # views (best effort)
        try:
            for s in el.find_elements(By.CSS_SELECTOR, "span[aria-label]"):
                lab = s.get_attribute("aria-label") or ""
                if "View" in lab:
                    vc = _num_from_label(lab)
                    if vc is not None:
                        row["view_count"] = vc
                        break
        except Exception:
            pass

        row["key"] = row["tweet_id"] or f"{row['user_handle']}|{row['timestamp_iso']}|{row['text'][:50]}"
        return row

    results, seen = [], set()
    not_growing, last_seen = 0, 0

    for _ in range(max_scrolls):
        articles = get_article_elements(driver)
        for art in articles:
            row = parse_article(art)
            if not (row["tweet_id"] or row["text"]):
                continue
            if row["key"] in seen:
                continue
            seen.add(row["key"])
            results.append(row)
            print(
                f"[{row['timestamp_iso']}] {row['user_display_name']} {row['user_handle']} "
                f"♥{row['like_count']} ↻{row['retweet_count']} 💬{row['reply_count']} 👁️{row['view_count']}\n"
                f"{row['text']}\n{row['permalink']}\n" + "—"*40
            )

        if len(seen) == last_seen:
            not_growing += 1
        else:
            not_growing = 0
        last_seen = len(seen)
        if not_growing >= 6:
            break

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_secs)

    print(f"\nCollected {len(results)} unique tweets.")
    return results

# >>> Use it like this (keep your URL exactly as you wrote it):
TARGET_URL = "https://x.com/search?q=มันฮวา&f=live"  # or ANY Twitter/X link
results = scrape_tweets(driver, TARGET_URL, max_scrolls=80, pause_secs=1.2)


# In[ ]:


# ── MAIN SCRAPE LOOP ──────────────────────────────────────────────────────────
for i in range(MAX_SCROLLS):
    articles = get_article_elements(driver)

    for art in articles:
        row = parse_article(art)
        # skip if nothing meaningful
        if not (row["tweet_id"] or row["text"]):
            continue
        if row["key"] in seen:
            continue
        seen.add(row["key"])
        results.append(row)

        print(
            f"[{row['timestamp_iso']}] {row['user_display_name']} {row['user_handle']} "
            f"♥{row['like_count']} ↻{row['retweet_count']} 💬{row['reply_count']} 👁️{row['view_count']}\n"
            f"{row['text']}\n{row['permalink']}\n" + "—"*40
        )

    # stop if we’re not discovering more after several scrolls
    if len(seen) == last_seen:
        not_growing += 1
    else:
        not_growing = 0
    last_seen = len(seen)
    if not_growing >= 6:
        break

    # some pages (status permalink) won't load more on scroll — exit early
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(PAUSE_SECS)

# results now holds all parsed tweets from the page/session
print(f"\nCollected {len(results)} unique tweets.")
# driver.quit()  # ← uncomment when you're done


# -----------------------------------

# In[17]:


# pip install beautifulsoup4 lxml

import re, time
from bs4 import BeautifulSoup
from selenium import webdriver

# ----------------- minimal helpers -----------------
def normalize_x_url(u: str) -> str:
    if not u: return "https://x.com"
    if u.startswith("/"): u = "https://x.com" + u
    return u.replace("https://twitter.com", "https://x.com")

def clean_url(url: str) -> str:
    return (url or "").split("?")[0].rstrip("/")

def digits(s: str) -> int:
    if not s: return 0
    m = re.search(r'(\d[\d,\.]+)', s)
    return int(m.group(1).replace(",", "").replace(".", "")) if m else 0

def get_article_htmls(driver):
    # one JS call → list of outerHTML strings (no WebElements held)
    return driver.execute_script("""
        return Array.from(document.querySelectorAll('article[role="article"]'))
                    .map(n => n.outerHTML);
    """) or []

def parse_article_html(html: str):
    soup = BeautifulSoup(html, "lxml")

    # text
    tt = soup.select_one('div[data-testid="tweetText"]')
    text = tt.get_text(" ", strip=True) if tt else ""

    # user
    name = handle = profile_url = ""
    u = soup.select_one('div[data-testid="User-Name"]')
    if u:
        for s in u.select("span"):
            t = s.get_text(strip=True)
            if t.startswith("@"): handle = t
            elif not name and t:   name = t
        a = u.select_one("a[href^='/']")
        if a:
            profile_url = a.get("href", "") or ""
            if profile_url.startswith("/"):
                profile_url = "https://x.com" + profile_url

    # permalink + id + timestamp
    a = soup.select_one("a[href*='/status/']")
    permalink = tid = iso = ""
    if a:
        href = a.get("href", "") or ""
        if href.startswith("/"): href = "https://x.com" + href
        permalink = clean_url(href)
        m = re.search(r"/status/(\d+)$", permalink); tid = m.group(1) if m else ""
        t = a.find("time"); iso = t["datetime"] if (t and t.has_attr("datetime")) else ""

    # counts
    reply_btn   = soup.select_one('div[data-testid="reply"]')
    retweet_btn = soup.select_one('div[data-testid="retweet"], div[data-testid="unretweet"]')
    like_btn    = soup.select_one('div[data-testid="like"], div[data-testid="unlike"]')
    views_span  = soup.select_one('a[href*="/analytics"] span[data-testid="app-text-transition-container"]') \
                or soup.select_one('a[href*="/status/"] span[data-testid="app-text-transition-container"]')

    replies  = digits(reply_btn.get("aria-label", "") if reply_btn else "")
    retweets = digits(retweet_btn.get("aria-label", "") if retweet_btn else "")
    likes    = digits(like_btn.get("aria-label", "") if like_btn else "")
    views    = digits(views_span.get_text(strip=True) if views_span else "")

    mentions = sorted(set(re.findall(r'@\w+', text)))
    quote_el = soup.select_one('div[data-testid="quoteTweet"], div[data-testid="card.wrapper"], div[role="blockquote"]')
    quote = quote_el.get_text(" ", strip=True) if quote_el else ""

    key = tid or permalink or f"{handle}|{iso}|{hash(text)}"
    return {
        "key": key, "tweet_id": tid, "permalink": permalink, "timestamp_iso": iso,
        "user_display_name": name, "user_handle": handle, "user_profile_url": profile_url,
        "text": text, "mentions": mentions, "reply_count": replies,
        "retweet_count": retweets, "like_count": likes, "view_count": views, "quoted_text": quote
    }

def scrape_x_page(start_url: str, max_scrolls=60, pause_secs=1.0):
    driver = webdriver.Chrome()
    try:
        driver.get(normalize_x_url(start_url))
        seen, results, stagnant = set(), [], 0
        last_seen = 0

        for _ in range(max_scrolls):
            for html in get_article_htmls(driver):
                row = parse_article_html(html)
                if not row["tweet_id"] and not row["text"]:  # skip non-tweets/cards
                    continue
                if row["key"] in seen:                       # de-dupe
                    continue
                seen.add(row["key"]); results.append(row)

                print(
                    f"[{row['timestamp_iso']}] {row['user_display_name']} {row['user_handle']} "
                    f"♥{row['like_count']} ↻{row['retweet_count']} 💬{row['reply_count']} 👁️{row['view_count']}\n"
                    f"{row['text']}\n{row['permalink']}\n" + "—"*40
                )

            stagnant = stagnant + 1 if len(seen) == last_seen else 0
            last_seen = len(seen)
            if stagnant >= 6: break

            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause_secs)

        return results
    finally:
        driver.quit()

# ----------------- run it -----------------
# Put ANY X URL here (absolute or relative). Example:
#   "https://x.com/search?q=-ข้าวมันไก่&f=live"
#   "/someuser/status/1978382002123419775"
#   "https://x.com/someuser"
results = scrape_x_page("https://x.com/search?q=-ข้าวมันไก่&f=live", max_scrolls=60, pause_secs=1.0)


# In[7]:


# -------- Helpers
def normalize_x_url(u: str) -> str:
    if not u:
        return "https://x.com"
    # Allow relative paths like "/user/status/...", or twitter.com host
    if u.startswith("/"):
        u = "https://x.com" + u
    u = u.replace("https://twitter.com", "https://x.com")
    return u

def clean_url(url: str) -> str:
    if not url: return ""
    return url.split("?")[0].rstrip("/")

def digits(s: str) -> int:
    if not s: return 0
    m = re.search(r'(\d[\d,\.]+)', s)
    return int(m.group(1).replace(",", "").replace(".", "")) if m else 0

def get_article_htmls(d):
    """Return fresh outerHTML strings for all tweet <article>s in one JS call."""
    try:
        return d.execute_script("""
            const nodes = Array.from(document.querySelectorAll('article[role="article"]'));
            return nodes.map(n => n.outerHTML);
        """) or []
    except Exception:
        return []

def parse_article_html(html: str):
    soup = BeautifulSoup(html, "lxml")

    # text
    text = ""
    tt = soup.select_one('div[data-testid="tweetText"]')
    if tt:
        text = tt.get_text(" ", strip=True)

    # user
    name = handle = profile_url = ""
    u = soup.select_one('div[data-testid="User-Name"]')
    if u:
        for s in u.select("span"):
            t = s.get_text(strip=True)
            if t.startswith("@"): handle = t
            elif not name and t:   name = t
        a = u.select_one("a[href^='/']")
        if a:
            profile_url = a.get("href", "") or ""
            if profile_url.startswith("/"):
                profile_url = "https://x.com" + profile_url

    # permalink + id + timestamp
    permalink = tid = iso = ""
    a = soup.select_one("a[href*='/status/']")
    if a:
        href = a.get("href", "") or ""
        if href.startswith("/"):
            href = "https://x.com" + href
        permalink = clean_url(href)
        m = re.search(r"/status/(\d+)$", permalink)
        tid = m.group(1) if m else ""
        t = a.find("time")
        if t and t.has_attr("datetime"):
            iso = t["datetime"]

    # counts
    reply_btn   = soup.select_one('div[data-testid="reply"]')
    retweet_btn = soup.select_one('div[data-testid="retweet"], div[data-testid="unretweet"]')
    like_btn    = soup.select_one('div[data-testid="like"], div[data-testid="unlike"]')
    views_span  = soup.select_one('a[href*="/analytics"] span[data-testid="app-text-transition-container"]') \
                   or soup.select_one('a[href*="/status/"] span[data-testid="app-text-transition-container"]')

    replies  = digits(reply_btn.get("aria-label", "") if reply_btn else "")
    retweets = digits(retweet_btn.get("aria-label", "") if retweet_btn else "")
    likes    = digits(like_btn.get("aria-label", "") if like_btn else "")
    views    = digits(views_span.get_text(strip=True) if views_span else "")

    # mentions + quote
    mentions = sorted(set(re.findall(r'@\w+', text)))
    quote = ""
    q = soup.select_one('div[data-testid="quoteTweet"], div[data-testid="card.wrapper"], div[role="blockquote"]')
    if q:
        quote = q.get_text(" ", strip=True)

    return {
        "tweet_id": tid,
        "key": tid or permalink or f"{handle}|{iso}|{hash(text)}",
        "permalink": permalink,
        "timestamp_iso": iso,
        "user_display_name": name,
        "user_handle": handle,
        "user_profile_url": profile_url,
        "text": text,
        "mentions": mentions,
        "reply_count": replies,
        "retweet_count": retweets,
        "like_count": likes,
        "view_count": views,
        "quoted_text": quote,
    }


# In[11]:


# pip install beautifulsoup4 lxml  # run once

import re, time, urllib.parse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

# =================== CONFIG ===================
START_URL = "https://x.com/search?q=-ข้าวมันไก่&f=live"  # <- put ANY X url here (relative or absolute)
MAX_SCROLLS = 60
PAUSE_SECS = 1.0
# ==============================================

# -------- Driver (add options if you want headless, custom UA, etc.)
driver = webdriver.Chrome()

# -------- Run
start = normalize_x_url(START_URL)
driver.get(start)

# Optional: small wait so first batch renders
try:
    WebDriverWait(driver, 10).until(lambda d: len(get_article_htmls(d)) > 0)
except Exception:
    pass

seen = set()
results = []
not_growing = 0
last_seen = 0


# In[12]:


results = []


# In[ ]:


driver.get("https://x.com/search?q=มันฮวา&f=live")
for i in range(MAX_SCROLLS):
    html_list = get_article_htmls(driver)

    for html in html_list:
        row = parse_article_html(html)
        if not row["tweet_id"] and not row["text"]:
            continue
        if row["key"] in seen:
            continue
        seen.add(row["key"])
        results.append(row)

        print(
            f"[{row['timestamp_iso']}] {row['user_display_name']} {row['user_handle']} "
            f"♥{row['like_count']} ↻{row['retweet_count']} 💬{row['reply_count']} 👁️{row['view_count']}\n"
            f"{row['text']}\n{row['permalink']}\n" + "—"*40
        )

    # stop if we’re not discovering more after several scrolls
    if len(seen) == last_seen:
        not_growing += 1
    else:
        not_growing = 0
    last_seen = len(seen)
    if not_growing >= 6:
        break

    # some pages (status permalink) won't load more on scroll — exit early
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(PAUSE_SECS)

# results now holds all parsed tweets from the page/session


# -----------------------------------------------------

# In[35]:


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time


# In[ ]:





# In[36]:


#driver = webdriver.Firefox(executable_path=r'./geckodriver.exe')
driver = webdriver.Chrome()


# In[ ]:


driver.get("https://x.com/search?q=มันฮวา&f=live")


# In[39]:


for i in range(10):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1)


# In[45]:


# pip install beautifulsoup4 lxml

import re, time
from bs4 import BeautifulSoup

def clean_url(url):
    if not url: return ""
    return url.split("?")[0].rstrip("/")

def digits(s):
    if not s: return 0
    m = re.search(r'(\d[\d,\.]+)', s)
    return int(m.group(1).replace(",", "").replace(".", "")) if m else 0

def parse_article_html(html):
    soup = BeautifulSoup(html, "lxml")

    # text
    text = ""
    tt = soup.select_one('div[data-testid="tweetText"]')
    if tt:
        text = tt.get_text(" ", strip=True)

    # user
    name = handle = profile_url = ""
    u = soup.select_one('div[data-testid="User-Name"]')
    if u:
        for s in u.select("span"):
            t = s.get_text(strip=True)
            if t.startswith("@"): handle = t
            elif not name and t:   name = t
        a = u.select_one("a[href^='/']")
        if a:
            profile_url = a.get("href", "") or ""
            if profile_url.startswith("/"):
                profile_url = "https://x.com" + profile_url

    # permalink + id + timestamp
    permalink = tid = iso = ""
    a = soup.select_one("a[href*='/status/']")
    if a:
        href = a.get("href", "") or ""
        if href.startswith("/"):
            href = "https://x.com" + href
        permalink = clean_url(href)
        m = re.search(r"/status/(\d+)$", permalink)
        tid = m.group(1) if m else ""
        t = a.find("time")
        if t and t.has_attr("datetime"):
            iso = t["datetime"]

    # counts
    reply_btn   = soup.select_one('div[data-testid="reply"]')
    retweet_btn = soup.select_one('div[data-testid="retweet"], div[data-testid="unretweet"]')
    like_btn    = soup.select_one('div[data-testid="like"], div[data-testid="unlike"]')
    views_span  = soup.select_one('a[href*="/analytics"] span[data-testid="app-text-transition-container"]') \
                   or soup.select_one('a[href*="/status/"] span[data-testid="app-text-transition-container"]')

    replies  = digits(reply_btn.get("aria-label", "") if reply_btn else "")
    retweets = digits(retweet_btn.get("aria-label", "") if retweet_btn else "")
    likes    = digits(like_btn.get("aria-label", "") if like_btn else "")
    views    = digits(views_span.get_text(strip=True) if views_span else "")

    # mentions + quote
    mentions = sorted(set(re.findall(r'@\w+', text)))
    quote = ""
    q = soup.select_one('div[data-testid="quoteTweet"], div[data-testid="card.wrapper"], div[role="blockquote"]')
    if q:
        quote = q.get_text(" ", strip=True)

    return {
        "tweet_id": tid,
        "key": tid or permalink or f"{handle}|{iso}|{hash(text)}",
        "permalink": permalink,
        "timestamp_iso": iso,
        "user_display_name": name,
        "user_handle": handle,
        "user_profile_url": profile_url,
        "text": text,
        "mentions": mentions,
        "reply_count": replies,
        "retweet_count": retweets,
        "like_count": likes,
        "view_count": views,
        "quoted_text": quote,
    }

def get_article_htmls(driver):
    # One JS call → array of outerHTML strings. No WebElement objects stored.
    try:
        return driver.execute_script("""
            const nodes = Array.from(document.querySelectorAll('article[role="article"]'));
            return nodes.map(n => n.outerHTML);
        """) or []
    except Exception:
        return []

# ------- main loop (no WebElements = no stale refs) -------
seen = set()
results = []

not_growing = 0
last_seen_count = 0

for _ in range(80):  # scroll attempts; tweak as needed
    html_list = get_article_htmls(driver)

    for html in html_list:
        row = parse_article_html(html)

        # skip obvious non-tweets
        if not row["tweet_id"] and not row["text"]:
            continue

        key = row["key"]
        if key in seen:
            continue
        seen.add(key)
        results.append(row)

        print(
            f"[{row['timestamp_iso']}] {row['user_display_name']} {row['user_handle']} "
            f"♥{row['like_count']} ↻{row['retweet_count']} 💬{row['reply_count']} 👁️{row['view_count']}\n"
            f"{row['text']}\n{row['permalink']}\n" + "—"*40
        )

    # stop if we’re not discovering anything new
    if len(seen) == last_seen_count:
        not_growing += 1
    else:
        not_growing = 0
    last_seen_count = len(seen)
    if not_growing >= 6:
        break

    # scroll + small pause
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1.0)


# In[46]:


# pip install beautifulsoup4 lxml  # (run once in your env)

import re, time
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException

def clean_url(url):
    if not url: 
        return ""
    return url.split("?")[0].rstrip("/")

def digits(s):
    if not s: return 0
    m = re.search(r'(\d[\d,\.]+)', s)
    return int(m.group(1).replace(",", "").replace(".", "")) if m else 0

def parse_article_html(html):
    soup = BeautifulSoup(html, "lxml")

    # text
    text = ""
    tt = soup.select_one('div[data-testid="tweetText"]')
    if tt:
        text = tt.get_text(" ", strip=True)

    # user (display + handle + profile link)
    name = handle = profile_url = ""
    u = soup.select_one('div[data-testid="User-Name"]')
    if u:
        for s in u.select("span"):
            t = s.get_text(strip=True)
            if t.startswith("@"): handle = t
            elif not name and t:   name = t
        a = u.select_one("a[href^='/']")
        if a:
            profile_url = a.get("href", "")
            if profile_url.startswith("/"):
                profile_url = "https://x.com" + profile_url  # <-- make absolute

    # permalink + id + timestamp
    permalink = tid = iso = ""
    a = soup.select_one("a[href*='/status/']")
    if a:
        href = a.get("href", "")
        if href.startswith("/"):
            href = "https://x.com" + href                     # <-- make absolute
        permalink = clean_url(href)                           # <-- then normalize
        m = re.search(r"/status/(\d+)$", permalink)
        tid = m.group(1) if m else ""
        t = a.find("time")
        if t and t.has_attr("datetime"):
            iso = t["datetime"]

    # counts
    reply_btn   = soup.select_one('div[data-testid="reply"]')
    retweet_btn = soup.select_one('div[data-testid="retweet"], div[data-testid="unretweet"]')
    like_btn    = soup.select_one('div[data-testid="like"], div[data-testid="unlike"]')
    views_span  = soup.select_one('a[href*="/analytics"] span[data-testid="app-text-transition-container"]') \
                   or soup.select_one('a[href*="/status/"] span[data-testid="app-text-transition-container"]')

    replies  = digits(reply_btn.get("aria-label", "") if reply_btn else "")
    retweets = digits(retweet_btn.get("aria-label", "") if retweet_btn else "")
    likes    = digits(like_btn.get("aria-label", "") if like_btn else "")
    views    = digits(views_span.get_text(strip=True) if views_span else "")

    # mentions + quote
    mentions = sorted(set(re.findall(r'@\w+', text)))
    quote = ""
    q = soup.select_one('div[data-testid="quoteTweet"], div[data-testid="card.wrapper"], div[role="blockquote"]')
    if q:
        quote = q.get_text(" ", strip=True)

    return {
        "tweet_id": tid,
        "key": tid or permalink or f"{handle}|{iso}|{hash(text)}",
        "permalink": permalink,
        "timestamp_iso": iso,
        "user_display_name": name,
        "user_handle": handle,
        "user_profile_url": profile_url,
        "text": text,
        "mentions": mentions,
        "reply_count": replies,
        "retweet_count": retweets,
        "like_count": likes,
        "view_count": views,
        "quoted_text": quote,
    }

# ------- use this loop -------
seen = set()
results = []

for _ in range(40):  # adjust number of scrolls
    # snapshot current articles
    try:
        arts = driver.find_elements(By.CSS_SELECTOR, 'article[role="article"]')
    except StaleElementReferenceException:
        arts = []

    for art in arts:
        # take a frozen HTML snapshot (won't go stale)
        try:
            html = driver.execute_script("return arguments[0].outerHTML;", art)
        except StaleElementReferenceException:
            continue  # it re-rendered mid-call; skip

        row = parse_article_html(html)

        # skip obvious non-tweets
        if not row["tweet_id"] and not row["text"]:
            continue

        # de-dupe across the whole session
        if row["key"] in seen:
            continue
        seen.add(row["key"])
        results.append(row)

        print(
            f"[{row['timestamp_iso']}] {row['user_display_name']} {row['user_handle']} "
            f"♥{row['like_count']} ↻{row['retweet_count']} 💬{row['reply_count']} 👁️{row['view_count']}\n"
            f"{row['text']}\n{row['permalink']}\n" + "—"*40
        )

    # scroll and let new items load
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1.0)


# In[47]:


import re, time
from selenium.webdriver.common.by import By

def clean_url(url):
    if not url: return ""
    url = url.split("?")[0].rstrip("/")
    return url

def number_from(el):
    if not el: return 0
    lab = el.get_attribute("aria-label") or ""
    txt = el.text or ""
    m = re.search(r'(\d[\d,\.]+)', lab) or re.search(r'(\d[\d,\.]+)', txt)
    return int(m.group(1).replace(",", "").replace(".", "")) if m else 0

def first(el, css):
    xs = el.find_elements(By.CSS_SELECTOR, css)
    return xs[0] if xs else None

def parse_article(a):
    # permalink & id
    permalink = ""; tid = ""; iso = ""
    try:
        a_link = a.find_element(By.CSS_SELECTOR, "a[href*='/status/']")
        permalink = clean_url(a_link.get_attribute("href") or "")
        m = re.search(r'/status/(\d+)$', permalink)
        tid = m.group(1) if m else ""
        try:
            iso = a_link.find_element(By.TAG_NAME, "time").get_attribute("datetime") or ""
        except Exception:
            pass
    except Exception:
        pass

    # user
    disp = handle = prof = ""
    try:
        user = a.find_element(By.CSS_SELECTOR, 'div[data-testid="User-Name"]')
        for s in user.find_elements(By.CSS_SELECTOR, "span"):
            t = s.text.strip()
            if t.startswith("@"): handle = t
            elif not disp and t:   disp = t
        link = first(user, "a[href^='/']")
        if link:
            prof = link.get_attribute("href") or ""
            if prof.startswith("/"): prof = "https://x.com" + prof
    except Exception:
        pass

    # text
    try:
        text = a.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]').text
    except Exception:
        text = ""

    # counts
    reply_btn   = first(a, 'div[data-testid="reply"]')
    retweet_btn = first(a, 'div[data-testid="retweet"], div[data-testid="unretweet"]')
    like_btn    = first(a, 'div[data-testid="like"], div[data-testid="unlike"]')
    views_el    = first(a, 'a[href*="/analytics"] span[data-testid="app-text-transition-container"]') \
                  or first(a, 'a[href*="/status/"] span[data-testid="app-text-transition-container"]')

    return {
        "tweet_id": tid,
        "key": tid or permalink or f"{handle}|{iso}|{hash(text)}",  # stable key
        "permalink": permalink,
        "timestamp_iso": iso,
        "user_display_name": disp,
        "user_handle": handle,
        "user_profile_url": prof,
        "text": text,
        "reply_count": number_from(reply_btn),
        "retweet_count": number_from(retweet_btn),
        "like_count": number_from(like_btn),
        "view_count": number_from(views_el),
    }

# ---- use this instead of your current 'find tweetText' block ----
seen = set()
results = []

for _ in range(30):  # scroll more/less as you like
    arts = driver.find_elements(By.CSS_SELECTOR, 'article[role="article"]')
    for art in arts:
        row = parse_article(art)

        # skip obvious non-tweets (no id+no text)
        if not row["tweet_id"] and not row["text"]:
            continue

        # de-dupe (handles re-renders and reinsertions)
        if row["key"] in seen:
            continue
        seen.add(row["key"])
        results.append(row)

        # print immediately (or store and print later)
        print(
            f"[{row['timestamp_iso']}] {row['user_display_name']} {row['user_handle']} "
            f"♥{row['like_count']} ↻{row['retweet_count']} 💬{row['reply_count']} 👁️{row['view_count']}\n"
            f"{row['text']}\n{row['permalink']}\n" + "—"*40
        )

    # scroll down once and let new items load
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1.0)


# In[58]:


import pandas as pd

csv_file = r"D:\Documents\University\Senior Project\gender-bias-detection\services\scraper\src\scraper\blog_based\tweets_data.csv"

# Load existing CSV
df = pd.read_csv(csv_file)

# Drop duplicates based on tweet ID or URL
df.drop_duplicates(subset=["id", "url"], inplace=True)

# Optionally reset index
df.reset_index(drop=True, inplace=True)

# Save cleaned CSV
df.to_csv(csv_file, index=False)

print(f"CSV cleaned. Total unique rows: {len(df)}")

