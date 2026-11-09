"""
Text preprocessing shared by TF-IDF and BM25 (Person 2 - Soni).

Both lexical retrievers MUST turn text into words in exactly the same way,
for documents and for queries. Otherwise "Dividends" in a query would never
match "dividend" in a passage.

    tokenize("What's a Roth IRA?")                         -> ['what', 's', 'a', 'roth', 'ira']
    tokenize("What's a Roth IRA?", remove_stopwords=True)  -> ['s', 'roth', 'ira']
    tokenize("running dividends", stem=True)               -> ['run', 'dividend']
"""

import re
from functools import lru_cache

# Words that appear in almost every text and carry little meaning.
# Kept as a fixed list (no download needed) so everyone gets identical results.
STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because been before being
below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down
during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here
here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves
out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the
their theirs them themselves then there there's these they they'd they'll they're they've this those
through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll
you're you've your yours yourself yourselves
s t d ll m re ve don doesn didn isn aren wasn weren hasn haven hadn won wouldn shouldn couldn mustn shan
""".split())

_WORD = re.compile(r"[a-z0-9]+")


_porter = None


@lru_cache(maxsize=None)  # remember results: the same word is stemmed thousands of times
def _stem(word: str) -> str:
    # Porter stemmer from NLTK (pip install nltk). Imported here so the
    # package is only needed when stemming is switched on.
    global _porter
    if _porter is None:
        from nltk.stem import PorterStemmer
        _porter = PorterStemmer()
    return _porter.stem(word)


def tokenize(text: str, remove_stopwords: bool = False, stem: bool = False) -> list[str]:
    """
    1. lowercase                 "Roth IRA?" -> "roth ira?"
    2. keep only letters/digits  "roth ira?" -> ["roth", "ira"]   (punctuation becomes a split point)
    3. optional: drop stopwords
    4. optional: stem words      "dividends" -> "dividend"
    """
    words = _WORD.findall(text.lower())
    if remove_stopwords:
        # Apostrophes were split off in step 2 ("don't" -> "don", "t"), so the
        # leftover pieces "s", "t", "ll" ... are in STOPWORDS as well.
        words = [w for w in words if w not in STOPWORDS]
    if stem:
        words = [_stem(w) for w in words]
    return words
