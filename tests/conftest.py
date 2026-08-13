import sys
import os

# Ensure root workspace directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

@pytest.fixture
def sample_short_text():
    return "Corporations are legal entities authorized to act as a single body."

@pytest.fixture
def sample_multi_sentence_text():
    return (
        "Rachel Carson wrote The Obligation to Endure in 1962. "
        "She argued that chemical pesticides harm the natural environment. "
        "Her work helped launch the modern environmental movement."
    )

@pytest.fixture
def sample_paragraph_text():
    return (
        "First paragraph describing corporate structure and governance principles.\n\n"
        "Second paragraph detailing environmental conservation efforts and Rachel Carson's books."
    )

@pytest.fixture
def sample_corpus():
    docs = [
        "A corporation is a company or group of people authorized to act as a single entity and recognized as such in law.",
        "Rachel Carson writes The Obligation to Endure because man tries to eliminate insects with pesticides.",
        "Honesty is the quality of being honest and having strong moral principles and integrity."
    ]
    metadatas = [
        {"doc_id": "1102432", "source": "MSMARCO-XI"},
        {"doc_id": "1102431", "source": "MSMARCO-XI"},
        {"doc_id": "205107", "source": "MSMARCO-XI"}
    ]
    return docs, metadatas
