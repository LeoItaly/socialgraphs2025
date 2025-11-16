
"""
Genre extractor for Wikipedia artist pages (infobox-based).

Usage example (in a notebook):
------------------------------
from genre_scraper import get_genres_for_artists, plot_top_genres, basic_stats

artists = ["The Rolling Stones", "The Beatles", "The Animals", "David Bowie"]
artist_to_genres = get_genres_for_artists(artists)

stats = basic_stats(artist_to_genres)
print(stats)

plot_top_genres(artist_to_genres, top_k=15)

Notes:
------
* Only assigns genres if an infobox row labeled "Genre" or "Genres" is found.
* All genres are normalized to lowercase; common rock-and-roll variants are merged.
* HTTP requests include a polite User-Agent and handle redirects.
* Add your own normalization rules in `normalize_genre` if you need them.
* Test odd cases like disambiguation pages and bands with unusual markup.
"""
import time
import re
from collections import Counter
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "SocialGraphs-GenreScraper/1.0 (course use; contact: your-email@example.com)"

# ---------- Utilities ----------

def _request_with_retries(params: Dict, max_retries: int = 3, backoff: float = 0.75) -> Optional["requests.Response"]:
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(WIKI_API_URL, params=params, headers=headers, timeout=15)
            if resp.status_code == 200:
                return resp
        except requests.RequestException:
            pass
        time.sleep(backoff * attempt)
    return None

def _fetch_page_html(title: str) -> Optional[str]:
    """
    Use MediaWiki 'parse' to get rendered HTML for the page title (follows redirects).
    Returns HTML (string) or None.
    """
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "redirects": 1,
        "prop": "text",
        "formatversion": 2,
    }
    resp = _request_with_retries(params)
    if not resp:
        return None
    data = resp.json()
    if "error" in data:
        return None
    return data.get("parse", {}).get("text", None)

def _extract_infobox_node(soup: BeautifulSoup):
    """
    Find the main infobox table node.
    Wikipedia often uses classes like 'infobox', 'infobox vcard', etc.
    """
    candidates = soup.select("table.infobox, table.infobox.vcard")
    return candidates[0] if candidates else None

def _text_cleanup(s: str) -> str:
    # Remove footnote markers like [1], [a], etc. Collapse spaces, unify hyphens.
    import re as _re
    s = _re.sub(r"\[\s*[0-9a-zA-Z]+\s*\]", "", s)
    s = s.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    s = _re.sub(r"\s+", " ", s).strip()
    return s

import re as _re
_ROCK_N_ROLL_RE = _re.compile(r"rock\s*(?:&|and|\+|n|n'|n’)?\s*roll", _re.I)
_AMP_RE = _re.compile(r"\s*&amp;\s*", _re.I)

def normalize_genre(g: str) -> str:
    g = _text_cleanup(g).lower()

    # Standardize ampersands in strings like "rhythm & blues"
    g = _AMP_RE.sub(" and ", g)

    # Merge common rock and roll variants
    if _ROCK_N_ROLL_RE.search(g):
        return "rock and roll"

    # Map a few frequent near-duplicates
    replacements = {
        "r&b": "rhythm and blues",
        "r and b": "rhythm and blues",
        "alt-rock": "alternative rock",
        "alt. rock": "alternative rock",
        "hip hop music": "hip hop",
        "electro-pop": "electropop",
    }
    if g in replacements:
        return replacements[g]

    return g

def _genres_from_infobox_html(infobox) -> List[str]:
    """
    Parse infobox table and extract the 'Genre' or 'Genres' row.
    Return a list of genre strings (deduplicated, normalized).
    """
    rows = infobox.select("tr")
    raw_genres: List[str] = []

    for tr in rows:
        th = tr.find("th")
        if not th:
            continue
        label = th.get_text(separator=" ").strip().lower()
        if label not in ("genre", "genres"):
            continue

        td = tr.find("td")
        if not td:
            continue

        # Prefer anchors inside list-like markup
        links = td.select("a")
        if links:
            for a in links:
                txt = a.get_text(separator=" ").strip()
                if txt:
                    raw_genres.append(txt)
        else:
            # fallback: take text content, split by common separators
            txt = td.get_text(separator="|")
            for piece in txt.split("|"):
                p = piece.strip()
                if p:
                    raw_genres.append(p)

    # Normalize + dedupe
    norm = [normalize_genre(g) for g in raw_genres]
    norm = [g for g in norm if g and g.lower() not in {"music", "musical group"}]
    seen = set()
    genres = []
    for g in norm:
        if g not in seen:
            genres.append(g)
            seen.add(g)
    return genres

def get_genres_for_artist(title: str) -> List[str]:
    html = _fetch_page_html(title)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    infobox = _extract_infobox_node(soup)
    if not infobox:
        return []
    return _genres_from_infobox_html(infobox)

def get_genres_for_artists(artists: List[str], sleep_s: float = 0.2) -> Dict[str, List[str]]:
    """
    Iterate through artist titles, fetching genres when present.
    Respects a small sleep between calls to be polite.
    """
    out: Dict[str, List[str]] = {}
    for name in artists:
        genres = get_genres_for_artist(name)
        if genres:
            out[name] = genres
        time.sleep(sleep_s)
    return out

# ---------- Reporting ----------

def basic_stats(artist_to_genres: Dict[str, List[str]]):
    """Return required stats as a small dict."""
    n_with_genres = len(artist_to_genres)
    if n_with_genres == 0:
        return {
            "n_nodes_with_genres": 0,
            "avg_genres_per_node": 0.0,
            "n_distinct_genres": 0,
        }
    avg_per = sum(len(v) for v in artist_to_genres.values()) / n_with_genres
    distinct = set(g for gs in artist_to_genres.values() for g in gs)
    return {
        "n_nodes_with_genres": n_with_genres,
        "avg_genres_per_node": float(avg_per),
        "n_distinct_genres": len(distinct),
    }

def top_genre_counts(artist_to_genres: Dict[str, List[str]], top_k: int = 15):
    c = Counter(g for gs in artist_to_genres.values() for g in gs)
    return c.most_common(top_k)

def plot_top_genres(artist_to_genres: Dict[str, List[str]], top_k: int = 15):
    """Plot a histogram (bar chart) for the top-k genres by artist counts."""
    counts = Counter(g for gs in artist_to_genres.values() for g in gs)
    top = counts.most_common(top_k)
    if not top:
        print("No genres to plot.")
        return

    labels = [g for g, _ in top]
    values = [c for _, c in top]

    plt.figure(dpi=120)
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.xlabel("Genre")
    plt.ylabel("Artist count")
    plt.title(f"Top {top_k} genres by artist count")
    plt.tight_layout()
    plt.show()

# ---------- Light tests ----------

def _sanity_tests():
    # Normalization
    assert normalize_genre("Rock & Roll") == "rock and roll"
    assert normalize_genre("R&B") == "rhythm and blues"
    assert normalize_genre("Electro-Pop") == "electropop"

    # Minimal HTML snippet test
    frag = """
    <table class="infobox vcard">
      <tr><th scope="row">Genres</th>
          <td><div class="hlist"><ul>
            <li><a>Rock</a></li>
            <li><a>Blues</a></li>
            <li><a>Rock &amp; Roll</a></li>
          </ul></div></td>
      </tr>
    </table>
    """
    soup = BeautifulSoup(frag, "html.parser")
    genres = _genres_from_infobox_html(soup)
    assert "rock" in genres and "blues" in genres and "rock and roll" in genres

if __name__ == "__main__":
    _sanity_tests()
    print("Sanity tests passed. Import this module in your notebook to run on your artist list.")
