import pytest

from src.retrievers.preprocessing import preprocess_text


def test_basic_tokenization():
    text = "The STOCK Market is Growing!"

    result = preprocess_text(text)

    assert result == ["the", "stock", "market", "is", "growing"]


def test_punctuation_removal():
    text = "Hello, World! How are you?"

    result = preprocess_text(text)

    assert result == ["hello", "world", "how", "are", "you"]


def test_stopword_removal():
    text = "The stock market is growing rapidly"

    result = preprocess_text(text, remove_stopwords=True)

    assert result == ["stock", "market", "growing", "rapidly"]


def test_stemming():
    text = "playing played plays running studies companies"

    result = preprocess_text(text, stemming=True)

    assert result == ["play", "play", "play", "run", "studi", "compani"]


def test_stopwords_and_stemming():
    text = "The companies are running quickly"

    result = preprocess_text(text, remove_stopwords=True, stemming=True)

    assert result == ["compani", "run", "quickli"]


def test_negations_are_preserved():
    text = "The market is not stable"

    result = preprocess_text(text, remove_stopwords=True)

    assert result == ["market", "not", "stable"]


def test_empty_string():
    assert preprocess_text("") == []


def test_whitespace():
    assert preprocess_text("   hello    world   ") == ["hello", "world"]


def test_invalid_input():
    with pytest.raises(TypeError):
        preprocess_text(123)
