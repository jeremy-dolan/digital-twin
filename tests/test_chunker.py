"""Tests for rag.chunk_curated_lines()"""

import pytest
from rag import chunk_curated_lines


BASIC_BIO = """\
# Section one
Fact one.
Fact two.

# Section two
Fact three.
"""

def test_basic_chunking():
    chunks = chunk_curated_lines(BASIC_BIO)
    assert len(chunks) == 3
    assert chunks[0].text == "Fact one."
    assert chunks[0].metadata == {'section': 'Section one', 'chunk': 1}
    assert chunks[2].metadata == {'section': 'Section two', 'chunk': 1}


def test_guidance_attached_to_preceding_chunk():
    text = """\
# Personal history
I was born in Los Angeles, California but grew up in Chicago.
    guidance: Only mention Los Angeles if the user asks specifically where.
"""
    chunks = chunk_curated_lines(text)
    assert len(chunks) == 1
    assert chunks[0].text == "I was born in Los Angeles, California but grew up in Chicago."
    assert chunks[0].metadata['guidance'] == "Only mention Los Angeles if the user asks specifically where."


def test_guidance_case_insensitive():
    text = """\
# Section
Some fact.
    Guidance: Use carefully.
"""
    chunks = chunk_curated_lines(text)
    assert chunks[0].metadata['guidance'] == "Use carefully."


def test_guidance_not_embedded_in_text():
    """Guidance lines should not appear in chunk text (which is what gets embedded)."""
    text = """\
# Section
Some fact.
    guidance: A hint.
"""
    chunks = chunk_curated_lines(text)
    assert 'guidance' not in chunks[0].text.lower()
    assert 'hint' not in chunks[0].text.lower()


def test_chunk_without_guidance_has_no_key():
    chunks = chunk_curated_lines(BASIC_BIO)
    assert 'guidance' not in chunks[0].metadata


def test_guidance_before_any_chunk_raises():
    text = """\
# Section
    guidance: Orphaned guidance.
Actual fact.
"""
    with pytest.raises(ValueError, match="Guidance before first chunk"):
        chunk_curated_lines(text)


def test_duplicate_guidance_raises():
    text = """\
# Section
Some fact.
    guidance: First hint.
    guidance: Second hint.
"""
    with pytest.raises(ValueError, match="Duplicate guidance"):
        chunk_curated_lines(text)


def test_empty_section_header_raises():
    with pytest.raises(ValueError, match="Empty section header"):
        chunk_curated_lines("#\nSome fact.\n")


def test_content_before_section_raises():
    with pytest.raises(ValueError, match="Content before first section header"):
        chunk_curated_lines("Orphaned fact.\n")


def test_duplicate_section_raises():
    text = """\
# Dupe
Fact.
# Dupe
Another fact.
"""
    with pytest.raises(ValueError, match="Duplicate section name"):
        chunk_curated_lines(text)


def test_multiple_chunks_only_last_gets_guidance():
    text = """\
# Section
Fact one.
Fact two.
    guidance: Applies to fact two only.
Fact three.
"""
    chunks = chunk_curated_lines(text)
    assert len(chunks) == 3
    assert 'guidance' not in chunks[0].metadata
    assert chunks[1].metadata['guidance'] == "Applies to fact two only."
    assert 'guidance' not in chunks[2].metadata


def test_skips_empty_lines():
    text = "# S\n\n\nFact.\n\n"
    chunks = chunk_curated_lines(text)
    assert len(chunks) == 1


def test_headers_not_included_as_chunks():
    text = "# Header\nContent."
    chunks = chunk_curated_lines(text)
    assert len(chunks) == 1
    assert chunks[0].text == "Content."


def test_multi_hash_header():
    text = "## Sub Header\nFact."
    chunks = chunk_curated_lines(text)
    assert chunks[0].metadata["section"] == "Sub Header"


def test_empty_input():
    assert chunk_curated_lines("") == []


def test_only_headers():
    assert chunk_curated_lines("# A\n# B\n# C") == []


def test_chunk_index_resets_per_section():
    text = "# A\nLine 1.\n# B\nLine 2."
    chunks = chunk_curated_lines(text)
    assert chunks[0].metadata["chunk"] == 1
    assert chunks[1].metadata["chunk"] == 1
