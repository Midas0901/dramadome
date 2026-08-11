from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import json
import ast
import re
from collections import Counter

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load data once at startup
# ---------------------------------------------------------------------------
dramas = pd.read_csv("data/dramas.csv")
users = pd.read_csv("data/users.csv")

dramas["genres"] = dramas["genres"].fillna("")
dramas["tags"] = dramas["tags"].fillna("")
dramas["content"] = dramas["content"].fillna("")
dramas["name"] = dramas["name"].fillna("")
dramas["no_of_viewers"] = pd.to_numeric(dramas["no_of_viewers"], errors="coerce").fillna(0)

dramas["genre_list"] = dramas["genres"].apply(
    lambda g: [x.strip() for x in g.split(",") if x.strip()]
)
dramas["tag_list"] = dramas["tags"].apply(
    lambda t: [x.strip() for x in t.split(",") if x.strip()]
)
dramas["main_role"] = dramas["main_role"].fillna("")
dramas["support_role"] = dramas["support_role"].fillna("")
dramas["cast_list"] = (dramas["main_role"] + ", " + dramas["support_role"]).apply(
    lambda c: [x.strip() for x in c.split(",") if x.strip()]
)
dramas["rating"] = pd.to_numeric(dramas["rating"], errors="coerce")

# ---- reviewer-gender-lean classification -----------------------------------
# reviewer_gender_info is a Counter-style string, e.g. "Counter({'female': 318, 'male': 44})"
def _parse_gender_counter(cell):
    try:
        return eval(str(cell), {"Counter": Counter, "__builtins__": {}})
    except Exception:
        return Counter()

def _female_ratio(cell):
    c = _parse_gender_counter(cell)
    f, m = c.get("female", 0), c.get("male", 0)
    total = f + m
    return (f / total) if total > 0 else np.nan

dramas["female_ratio"] = dramas["reviewer_gender_info"].apply(_female_ratio)
_valid_ratios = dramas["female_ratio"].dropna()
# Thresholds derived from the data's own distribution (it's heavily female-skewed
# overall, so fixed 50% cutoffs would barely split anything).
_P20 = _valid_ratios.quantile(0.20) if len(_valid_ratios) else 0.7
_P80 = _valid_ratios.quantile(0.80) if len(_valid_ratios) else 0.95

def _audience_lean(ratio):
    if pd.isna(ratio):
        return "Unknown"
    if ratio <= _P20:
        return "More Male Following"
    if ratio >= _P80:
        return "Overwhelmingly Female Audience"
    return "Mixed Audience"

dramas["audience_lean"] = dramas["female_ratio"].apply(_audience_lean)

# ---- top billed actors/actresses (for the actor browse feature) ------------
_actor_counts = Counter()
for cell in dramas["main_role"]:
    for a in cell.split(","):
        a = a.strip()
        if a:
            _actor_counts[a] += 1
TOP_ACTORS = [a for a, _ in _actor_counts.most_common(18)]

# Precompute a single lowercase "search blob" per drama (name + genres + tags + synopsis)
dramas["search_blob"] = (
    dramas["name"].str.lower() + " " +
    dramas["genres"].str.lower() + " " +
    dramas["tags"].str.lower() + " " +
    dramas["content"].str.lower()
)

# Convert user_watch_history from string back into a list
def parse_history(x):
    try:
        return json.loads(x)
    except Exception:
        try:
            return ast.literal_eval(x)
        except Exception:
            return []

users["user_watch_history"] = users["user_watch_history"].apply(parse_history)

# All genres available in the catalogue, ranked by how many titles have them
GENRE_COUNTS = Counter(g for gl in dramas["genre_list"] for g in gl)
ALL_GENRES = [g for g, _ in GENRE_COUNTS.most_common()]

# A curated shortlist of "category buttons" shown on the home page.
# Falls back gracefully if a genre isn't present in the dataset.
CATEGORY_BUTTONS = [
    "Action", "Adventure", "Romance", "Comedy", "Drama", "Family",
    "Fantasy", "Thriller", "Mystery", "Historical", "Sci-Fi", "Horror",
]
CATEGORY_BUTTONS = [c for c in CATEGORY_BUTTONS if c in GENRE_COUNTS] or ALL_GENRES[:12]

# ---------------------------------------------------------------------------
# Lightweight "semantic" search: expand a free-text query with related
# keywords so natural phrases (e.g. "super hero", "detective", "high school
# love") surface relevant titles even if that exact phrase never appears in
# the catalogue text. This is a keyword-expansion approach, not a true
# embedding model, but it gives noticeably better results than plain
# substring search.
# ---------------------------------------------------------------------------
SYNONYM_MAP = {
    "superhero": ["hero", "supernatural", "fantasy", "action", "power", "save the world"],
    "super hero": ["hero", "supernatural", "fantasy", "action", "power", "save the world"],
    "hero": ["hero", "supernatural", "fantasy", "action"],
    "spy": ["spy", "agent", "undercover", "thriller", "action", "crime"],
    "detective": ["detective", "crime", "mystery", "investigation", "police"],
    "police": ["police", "crime", "detective", "investigation", "law"],
    "murder": ["murder", "crime", "mystery", "thriller", "psychological"],
    "ghost": ["ghost", "supernatural", "horror", "fantasy", "spirit"],
    "vampire": ["vampire", "supernatural", "fantasy", "horror"],
    "time travel": ["time travel", "time slip", "fantasy", "sci-fi"],
    "zombie": ["zombie", "horror", "thriller", "apocalypse"],
    "high school": ["high school", "youth", "school", "teen"],
    "school": ["school", "youth", "high school", "teen"],
    "office": ["office", "workplace", "business", "career"],
    "workplace": ["office", "workplace", "business", "career"],
    "doctor": ["doctor", "medical", "hospital"],
    "hospital": ["hospital", "medical", "doctor"],
    "lawyer": ["lawyer", "law", "legal", "court"],
    "law": ["law", "legal", "lawyer", "court"],
    "king": ["king", "royal", "historical", "palace", "dynasty"],
    "queen": ["queen", "royal", "historical", "palace", "dynasty"],
    "royal": ["royal", "king", "queen", "historical", "palace", "dynasty"],
    "war": ["war", "military", "historical"],
    "military": ["military", "war", "historical"],
    "cooking": ["cooking", "food", "chef", "restaurant"],
    "chef": ["chef", "cooking", "food", "restaurant"],
    "idol": ["idol", "music", "singer", "entertainment industry"],
    "singer": ["singer", "music", "idol"],
    "sports": ["sports", "athlete", "team", "competition"],
    "martial arts": ["martial arts", "wuxia", "action", "kung fu"],
    "kung fu": ["martial arts", "wuxia", "action", "kung fu"],
    "wuxia": ["wuxia", "martial arts", "historical", "action"],
    "love triangle": ["love triangle", "romance", "melodrama"],
    "first love": ["first love", "youth", "romance", "school"],
    "revenge": ["revenge", "thriller", "psychological", "crime"],
    "second lead": ["romance", "melodrama", "love triangle"],
    "cute": ["comedy", "romance", "youth", "family"],
    "funny": ["comedy"],
    "sad": ["melodrama", "tearjerker", "tragedy"],
    "crying": ["melodrama", "tearjerker", "tragedy"],
    "action packed": ["action", "thriller", "crime"],
    "family drama": ["family", "life", "drama"],
    "reincarnation": ["reincarnation", "fantasy", "supernatural"],
    "alien": ["sci-fi", "alien", "fantasy"],
    "robot": ["sci-fi", "robot", "tokusatsu"],
}

STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "with", "like", "some", "movie",
    "movies", "series", "show", "shows", "drama", "dramas", "related",
    "about", "find", "me", "something", "that", "is", "in", "to", "for",
}


def expand_query(raw_query: str):
    """Turn free text into a list of weighted keyword terms."""
    q = raw_query.lower().strip()
    terms = []

    # Multi-word synonym phrases first (so "super hero" beats "hero")
    for phrase, expansions in SYNONYM_MAP.items():
        if phrase in q:
            terms.extend(expansions)

    # Then individual words, skipping stopwords and very short tokens
    words = re.findall(r"[a-z']+", q)
    for w in words:
        if w not in STOPWORDS and len(w) > 2:
            terms.append(w)

    # de-duplicate while preserving order
    seen = set()
    ordered = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def score_dramas_by_query(query: str, limit: int = 20):
    terms = expand_query(query)
    if not terms:
        return dramas.iloc[0:0]

    def row_score(blob, genre_list, tag_list, name):
        score = 0
        for t in terms:
            if t in name:
                score += 5          # title match is strongest signal
            if any(t == g.lower() for g in genre_list):
                score += 3          # exact genre match
            if any(t == g.lower() for g in tag_list):
                score += 3          # exact tag match
            if t in blob:
                score += 1          # loose match anywhere in text
        return score

    scores = [
        row_score(blob, gl, tl, name)
        for blob, gl, tl, name in zip(
            dramas["search_blob"], dramas["genre_list"], dramas["tag_list"], dramas["name"].str.lower()
        )
    ]
    scored = dramas.assign(_score=scores)
    scored = scored[scored["_score"] > 0]
    scored = scored.sort_values(["_score", "no_of_viewers"], ascending=[False, False])
    return scored.head(limit)


def to_cards(df: pd.DataFrame, extra_cols=None):
    """Trim a dramas dataframe down to the fields the templates need."""
    cols = ["name", "country", "genres", "no_of_viewers", "rating", "content"]
    if extra_cols:
        cols = cols + [c for c in extra_cols if c not in cols]
    out = df[cols].copy()
    out["content"] = out["content"].apply(
        lambda c: (c[:220] + "...") if isinstance(c, str) and len(c) > 220 else c
    )
    out["rating"] = out["rating"].apply(lambda r: None if pd.isna(r) else round(float(r), 1))
    return out.to_dict(orient="records")


def apply_sort(df: pd.DataFrame, sort_key: str):
    """Shared sort control used across category / search / actor / audience pages."""
    if sort_key == "rating_asc":
        return df.sort_values("rating", ascending=True, na_position="last")
    if sort_key == "rating_desc":
        return df.sort_values("rating", ascending=False, na_position="last")
    if sort_key == "viewers_asc":
        return df.sort_values("no_of_viewers", ascending=True)
    # default: viewers_desc
    return df.sort_values("no_of_viewers", ascending=False)


SORT_OPTIONS = [
    ("viewers_desc", "Most viewers"),
    ("viewers_asc", "Fewest viewers"),
    ("rating_desc", "Rating: High to Low"),
    ("rating_asc", "Rating: Low to High"),
]

AUDIENCE_CATEGORIES = [
    ("women", "Overwhelmingly Female Audience"),
    ("mixed", "Mixed Audience"),
    ("men", "More Male Following"),
]
AUDIENCE_SLUG_TO_LABEL = dict(AUDIENCE_CATEGORIES)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    sample_user_ids = users["user_id"].head(30).tolist()
    return render_template(
        "index.html",
        total_users=len(users),
        total_dramas=len(dramas),
        categories=CATEGORY_BUTTONS,
        all_genres=ALL_GENRES,
        countries=sorted(dramas["country"].dropna().unique().tolist()),
        top_actors=TOP_ACTORS,
        audience_categories=AUDIENCE_CATEGORIES,
        sample_user_ids=sample_user_ids,
    )


@app.route("/category/<genre>")
def category(genre):
    sort_key = request.args.get("sort", "viewers_desc")
    filtered = dramas[dramas["genre_list"].apply(lambda gl: genre.lower() in [g.lower() for g in gl])]
    filtered = apply_sort(filtered, sort_key).head(20)
    return render_template(
        "results.html",
        heading=f'Category: {genre}',
        subheading=f"{len(filtered)} title(s) tagged \u201c{genre}\u201d",
        results=to_cards(filtered),
        categories=CATEGORY_BUTTONS,
        active_category=genre,
        query=None,
        sort_options=SORT_OPTIONS,
        active_sort=sort_key,
        base_url=url_for("category", genre=genre),
    )


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    sort_key = request.args.get("sort", "")  # empty = relevance order
    if not query:
        return render_template(
            "results.html",
            heading="Search",
            subheading="Type something above to search the catalogue.",
            results=[],
            categories=CATEGORY_BUTTONS,
            active_category=None,
            query=query,
            sort_options=SORT_OPTIONS,
            active_sort=sort_key,
            base_url=url_for("search"),
        )
    matched = score_dramas_by_query(query, limit=30)
    if sort_key:
        matched = apply_sort(matched, sort_key)
    matched = matched.head(20)
    return render_template(
        "results.html",
        heading=f'Search results for "{query}"',
        subheading=f"{len(matched)} title(s) found",
        results=to_cards(matched),
        categories=CATEGORY_BUTTONS,
        active_category=None,
        query=query,
        sort_options=SORT_OPTIONS,
        active_sort=sort_key,
        base_url=url_for("search") + f"?q={query}",
    )


@app.route("/actor")
def actor_search():
    q = request.args.get("q", "").strip()
    sort_key = request.args.get("sort", "viewers_desc")
    if not q:
        return render_template(
            "results.html",
            heading="Browse by Actor / Actress",
            subheading="Type a name above, or pick one of the frequently-billed leads.",
            results=[],
            categories=CATEGORY_BUTTONS,
            active_category=None,
            query=None,
            sort_options=SORT_OPTIONS,
            active_sort=sort_key,
            base_url=url_for("actor_search"),
            top_actors=TOP_ACTORS,
        )
    ql = q.lower()
    filtered = dramas[dramas["cast_list"].apply(lambda cl: any(ql in a.lower() for a in cl))]
    filtered = apply_sort(filtered, sort_key).head(20)
    return render_template(
        "results.html",
        heading=f'Titles featuring "{q}"',
        subheading=f"{len(filtered)} title(s) found",
        results=to_cards(filtered),
        categories=CATEGORY_BUTTONS,
        active_category=None,
        query=None,
        sort_options=SORT_OPTIONS,
        active_sort=sort_key,
        base_url=url_for("actor_search") + f"?q={q}",
        top_actors=TOP_ACTORS,
    )


@app.route("/audience/<slug>")
def audience(slug):
    label = AUDIENCE_SLUG_TO_LABEL.get(slug)
    if not label:
        return "Unknown audience category", 404
    sort_key = request.args.get("sort", "viewers_desc")
    filtered = dramas[dramas["audience_lean"] == label]
    filtered = apply_sort(filtered, sort_key).head(20)
    return render_template(
        "results.html",
        heading=f"Audience: {label}",
        subheading=f"{len(filtered)} title(s), based on the reviewer gender split for each show",
        results=to_cards(filtered),
        categories=CATEGORY_BUTTONS,
        active_category=None,
        query=None,
        sort_options=SORT_OPTIONS,
        active_sort=sort_key,
        base_url=url_for("audience", slug=slug),
        audience_categories=AUDIENCE_CATEGORIES,
        active_audience=slug,
    )


@app.route("/for-you", methods=["GET", "POST"])
def for_you():
    """Personalized hybrid recommendation: combines a user's structured profile
    (preferred genre / country / watch history) with a free-text mood query,
    and returns an explainable top-5, per the client's success criteria."""
    sample_user_ids = users["user_id"].head(30).tolist()
    if request.method != "POST":
        return render_template(
            "for_you.html",
            sample_user_ids=sample_user_ids,
            user=None, results=None, selected_user_id=None, query=None,
        )

    user_id = request.form.get("user_id", "").strip()
    query = request.form.get("query", "").strip()

    user_row = users[users["user_id"] == user_id]
    if user_row.empty:
        return render_template(
            "for_you.html",
            sample_user_ids=sample_user_ids,
            user=None, results=None, selected_user_id=user_id, query=query,
            error=f'No user found with id "{user_id}".',
        )

    user_data = user_row.iloc[0].to_dict()
    pref_genre = str(user_data.get("preferred_genre", "")).lower()
    pref_country = str(user_data.get("preferred_country", ""))
    watched = set(t.lower() for t in user_data.get("user_watch_history", []))

    text_terms = expand_query(query) if query else []

    def score_row(row):
        score = 0.0
        reasons = []
        genre_list_lower = [g.lower() for g in row["genre_list"]]
        if pref_genre and pref_genre in genre_list_lower:
            score += 4
            reasons.append(f"Matches your preferred genre ({user_data['preferred_genre']})")
        if pref_country and row["country"] == pref_country:
            score += 2
            reasons.append(f"From your preferred country ({pref_country})")
        if row["name"].lower() in watched:
            return -1, []  # already watched -> exclude
        if text_terms:
            blob = row["search_blob"]
            hits = [t for t in text_terms if t in blob or t in genre_list_lower or t in [g.lower() for g in row["tag_list"]]]
            if hits:
                score += min(len(hits), 4)
                shown = ", ".join(sorted(set(hits))[:3])
                reasons.append(f'Matches your search ("{shown}")')
        # small popularity nudge so ties favor well-known titles
        score += min(row["no_of_viewers"] / 200000, 0.5)
        return score, reasons

    scores, reasons_list = [], []
    for _, row in dramas.iterrows():
        s, r = score_row(row)
        scores.append(s)
        reasons_list.append(r)

    scored = dramas.assign(_score=scores, _reasons=reasons_list)
    scored = scored[scored["_score"] > 0].sort_values("_score", ascending=False).head(5)

    cards = to_cards(scored)
    for card, reasons in zip(cards, scored["_reasons"].tolist()):
        card["reasons"] = reasons or ["Popular pick that fits your general profile"]

    return render_template(
        "for_you.html",
        sample_user_ids=sample_user_ids,
        user=user_data,
        results=cards,
        selected_user_id=user_id,
        query=query,
        error=None,
    )


@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    if request.method == "POST":
        genre = request.form.get("genre")
        country = request.form.get("country")

        filtered = dramas.copy()
        if genre and genre != "all":
            filtered = filtered[filtered["genres"].str.contains(genre, case=False, na=False)]
        if country and country != "all":
            filtered = filtered[filtered["country"] == country]

        filtered = filtered.sort_values("no_of_viewers", ascending=False).head(20)

        return render_template(
            "recommend.html",
            results=to_cards(filtered),
            selected_genre=genre,
            selected_country=country,
            genres=ALL_GENRES,
            countries=sorted(dramas["country"].dropna().unique().tolist()),
        )

    return render_template(
        "recommend.html",
        results=[],
        selected_genre=None,
        selected_country=None,
        genres=ALL_GENRES,
        countries=sorted(dramas["country"].dropna().unique().tolist()),
    )


@app.route("/user/<user_id>")
def user_detail(user_id):
    user = users[users["user_id"] == user_id]
    if user.empty:
        return "User not found", 404

    user_data = user.iloc[0].to_dict()
    history = user_data["user_watch_history"]

    watched = dramas[dramas["name"].isin(history)]

    return render_template(
        "user.html",
        user=user_data,
        watched=to_cards(watched),
    )


@app.route("/users")
def user_list():
    sample_users = users[["user_id", "gender", "location", "preferred_genre", "preferred_country"]].head(50)
    return render_template("users.html", users=sample_users.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True)
