from data import _build_fiqa_main_corpus


def test_fiqa_course_subset_uses_ground_truth_passages_and_course_ids():
    dataset = {
        "train": [
            {"question": "train question 1", "ground_truths": ["answer A", "answer B"]},
            {"question": "train question 2", "ground_truths": ["answer C"]},
        ],
        "validation": [
            {"question": "validation question", "ground_truths": ["answer D"]}
        ],
        "test": [
            {"question": "test question", "ground_truths": ["answer E", "answer E"]}
        ],
    }

    corpus, split_data = _build_fiqa_main_corpus(dataset)

    # Each ground-truth entry becomes its own document, even when text repeats.
    assert list(corpus) == [f"doc{i}" for i in range(6)]
    assert len(corpus) == 6
    assert corpus["doc0"] == "answer A"
    assert corpus["doc5"] == "answer E"

    # Query and document IDs continue across train -> validation -> test,
    # matching the course Corpus.__assign_ids implementation.
    assert list(split_data["train"]["queries"]) == ["q0", "q1"]
    assert list(split_data["dev"]["queries"]) == ["q2"]
    assert list(split_data["test"]["queries"]) == ["q3"]

    assert split_data["train"]["qrels"] == {
        "q0": {"doc0": 1, "doc1": 1},
        "q1": {"doc2": 1},
    }
    assert split_data["dev"]["qrels"] == {"q2": {"doc3": 1}}
    assert split_data["test"]["qrels"] == {"q3": {"doc4": 1, "doc5": 1}}

