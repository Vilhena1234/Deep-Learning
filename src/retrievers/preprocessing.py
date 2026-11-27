import re
from nltk.stem import PorterStemmer

# Common English stopwords.
# Negations such as "not" and "no" are intentionally preserved.
ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "while",
        "with",
        "without",
        "to",
        "from",
        "in",
        "on",
        "at",
        "for",
        "of",
        "by",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "can",
        "could",
        "should",
        "would",
        "will",
        "may",
        "might",
        "must",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "we",
        "they",
        "he",
        "she",
        "i",
        "you",
        "your",
        "my",
        "our",
        "their",
        "his",
        "her",
        "them",
        "there",
        "here",
        "then",
        "than",
        "so",
        "very",
        "just",
        "also",
        "into",
        "about",
        "over",
        "under",
        "up",
        "down",
        "out",
        "more",
        "most",
        "some",
        "any",
        "each",
        "such",
        "only",
        "own",
        "same",
        "too",
    }
)


# Initialise the stemmer once and reuse it.
STEMMER = PorterStemmer()


def preprocess_text(
    text: str, remove_stopwords: bool = False, stemming: bool = False
) -> list[str]:
    """
    Convert raw text into a list of processed tokens.

    Steps:
        1. Lowercase text.
        2. Remove punctuation.
        3. Tokenize text.
        4. Optionally remove stopwords.
        5. Optionally apply stemming.

    Args:
        text: Original input text.
        remove_stopwords: Whether to remove common English words.
        stemming: Whether to apply Porter stemming.

    Returns:
        A list of processed tokens.
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove punctuation
    text = re.sub(r"[^\w\s]|_", " ", text)

    # Step 3: Tokenization
    tokens = text.split()

    # Step 4: Optional stopword removal
    if remove_stopwords:
        tokens = [token for token in tokens if token not in ENGLISH_STOPWORDS]

    # Step 5: Optional stemming
    if stemming:
        tokens = [STEMMER.stem(token) for token in tokens]

    return tokens
