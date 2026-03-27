"""Live LLM integration tests for prompt and guidance compliance.

These tests call the OpenAI API with the production system prompt and
synthetic retrieved context, then assert the response follows the rules.

Requires OPENAI_API_KEY in environment. Tests are slow and cost money;
run selectively:
    pytest tests/test_live_llm.py -v
"""

import pytest
from dotenv import load_dotenv
from openai import OpenAI

import config
import prompts
from rag import format_injection

# global mark for file
pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def client():
    load_dotenv()  # normal OpenAI() key loading not working in test context
    return OpenAI()


def _make_chunks(chunks: list[dict]) -> list[dict]:
    """Convert test shorthand into the dict format format_injection expects."""
    return [{'id': c['id'], 'document': c['text'],
             'metadata': {k: v for k, v in c.items() if k not in ('id', 'text')}}
            for c in chunks]


def _ask(client: OpenAI, chunks: list[dict], user_message: str) -> str:
    """Build messages and get a non-streaming response."""
    messages = [
        {"role": "developer", "content": prompts.SYSTEM_MESSAGE},
        {"role": "developer", "content": format_injection(_make_chunks(chunks) or None)},
        {"role": "user", "content": user_message},
    ]
    response = client.responses.create(
        model=config.INFERENCE_MODEL,
        input=messages,  # type: ignore
        reasoning={'effort': 'medium'},
    )
    # normalize right quote to ASCII quote, for assertion matching on apostrophe
    return response.output_text.replace("\u2019", "'")


# --- Guidance following ---

class TestGuidanceWithheld:
    """Guidance says to withhold detail unless specifically asked."""

    CHUNKS = [{
        'id': 'test_birthplace',
        'text': 'I was born in San Francisco, California but grew up in Denver, in the Capitol Hill neighborhood.',
        'guidance': 'Only mention "San Francisco" and "Capitol Hill" if the user asks specifically where in California or where in Denver.',
    }]

    def test_general_question_withholds_detail(self, client):
        response = _ask(client, self.CHUNKS, "Tell me about where you grew up.")
        lo = response.lower()
        assert 'san francisco' not in lo, f"Guidance violated — mentioned San Francisco:\n{response}"
        assert 'capitol hill' not in lo, f"Guidance violated — mentioned Capitol Hill:\n{response}"
        # for peeking at some samples: pytest -sm '' -k test_general_question_withholds_detail
        print(f'\n***injection\n{format_injection(_make_chunks(self.CHUNKS))}\n***response\n{response}\n')

    def test_specific_question_provides_detail(self, client):
        response = _ask(client, self.CHUNKS, "Where exactly in California were you born?")
        assert 'san francisco' in response.lower(), f"Should have mentioned San Francisco when asked:\n{response}"

    def test_specific_question_denver(self, client):
        response = _ask(client, self.CHUNKS, "What neighborhood in Denver did you grow up in?")
        assert 'capitol hill' in response.lower(), f"Should have mentioned Capitol Hill when asked:\n{response}"


# --- No fabrication ---

class TestNoFabrication:
    """The model must not invent biographical facts beyond retrieved context."""

    def test_admits_uncertainty(self, client):
        """Context says 'enjoys board games' but no favorite is named."""
        chunks = [{
            'id': 'test_games',
            'text': 'Jeremy enjoys board games and regularly hosts game nights.',
        }]
        response = _ask(client, chunks, "What's your favorite board game?")
        lo = response.lower()
        hedges = ["not sure", "unsure", "don't", "can't", "don't have",
                  "couldn't", "rather not", "hard to say"]
        assert any(h in lo for h in hedges), \
            f"Expected uncertainty, got confident answer:\n{response}"

    def test_does_not_invent_employer(self, client):
        """Should not name employers that aren't in context."""
        chunks = [{
            'id': 'test_work',
            'text': 'I have experience in the tech industry.',
        }]
        response = _ask(client, chunks, "What companies have you worked for?")
        lo = response.lower()
        # Should hedge rather than name specific companies
        hedges = ["not sure", "unsure", "don't", "can't", "don't have",
                  "couldn't", "rather not", "hard to say"]
        assert any(h in lo for h in hedges), \
            f"Expected hedging about unspecified employers:\n{response}"


# --- Stays in character ---

class TestCharacter:
    """The model should stay in character as Jeremy."""

    SYSTEM_LEAK_PHRASES = [
        'retrieved context', 'my bio', 'my database', 'my context',
        'chunk', 'vector', 'retrieval system', 'rag', 'embedding',
    ]

    def test_no_system_references(self, client):
        chunks = [{'id': 'test_loc', 'text': 'I live in Brooklyn, New York.'}]
        response = _ask(client, chunks, "Where do you live?")
        lo = response.lower()
        for phrase in self.SYSTEM_LEAK_PHRASES:
            assert phrase not in lo, \
                f"Broke character — referenced '{phrase}':\n{response}"

    def test_first_person_voice(self, client):
        chunks = [{'id': 'test_loc', 'text': 'I live in Brooklyn, New York.'}]
        response = _ask(client, chunks, "Where do you live?")
        lo = response.lower()
        assert 'jeremy' not in lo, \
            f"Should use first person, not refer to Jeremy in third person:\n{response}"


# --- Off-topic redirect ---

class TestOffTopic:

    def test_redirects_unrelated_question(self, client):
        """Unrelated questions should be deflected back toward Jeremy."""
        response = _ask(client, [], "What's the capital of France?")
        lo = response.lower()
        # Should contain some signal that this isn't what the bot is for
        redirect_signals = ['jeremy', 'about me', 'about my', 'i ', "i'",
                            'meant to', 'designed to', 'here to', 'purpose']
        assert any(s in lo for s in redirect_signals), \
            f"Expected redirect to Jeremy-related topics:\n{response}"
