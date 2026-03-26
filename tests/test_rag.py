from unittest.mock import MagicMock

import pytest

from rag import chunk_curated_lines, build_context_injection


class TestChunkCuratedLines:

    def test_basic_chunking(self):
        text = "# Section A\nFact one.\nFact two.\n\n# Section B\nFact three."
        chunks = chunk_curated_lines(text)
        assert len(chunks) == 3
        assert chunks[0].text == "Fact one."
        assert chunks[1].text == "Fact two."
        assert chunks[2].text == "Fact three."

    def test_section_metadata(self):
        text = "# My Section\nFirst line.\nSecond line."
        chunks = chunk_curated_lines(text)
        assert chunks[0].metadata == {"section": "My Section", "chunk": 1}
        assert chunks[1].metadata == {"section": "My Section", "chunk": 2}

    def test_chunk_index_resets_per_section(self):
        text = "# A\nLine 1.\n# B\nLine 2."
        chunks = chunk_curated_lines(text)
        assert chunks[0].metadata["chunk"] == 1
        assert chunks[1].metadata["chunk"] == 1

    def test_skips_empty_lines(self):
        text = "# S\n\n\nFact.\n\n"
        chunks = chunk_curated_lines(text)
        assert len(chunks) == 1

    def test_headers_not_included_as_chunks(self):
        text = "# Header\nContent."
        chunks = chunk_curated_lines(text)
        assert len(chunks) == 1
        assert chunks[0].text == "Content."

    def test_multi_hash_header(self):
        text = "## Sub Header\nFact."
        chunks = chunk_curated_lines(text)
        assert chunks[0].metadata["section"] == "Sub Header"

    def test_empty_input(self):
        assert chunk_curated_lines("") == []

    def test_only_headers(self):
        assert chunk_curated_lines("# A\n# B\n# C") == []

    def test_duplicate_section_name_raises(self):
        text = "# Same\nFact one.\n# Same\nFact two."
        with pytest.raises(ValueError, match="Duplicate section name"):
            chunk_curated_lines(text)

    def test_content_before_any_header_raises(self):
        text = "Orphan line.\n# First\nFact."
        with pytest.raises(ValueError, match="Content before first section header"):
            chunk_curated_lines(text)

    def test_empty_section_header_raises(self):
        text = "# Valid\nFact.\n#\nMore."
        with pytest.raises(ValueError, match="Empty section header"):
            chunk_curated_lines(text)


class TestBuildContextInjection:
    """Test the retrieval filtering and formatting logic by mocking the OpenAI
    embedding call and ChromaDB collection.query() return value."""

    @pytest.fixture
    def mock_oai_client(self):
        client = MagicMock()
        # embed_strings calls client.embeddings.create; return a dummy embedding
        mock_response = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.0] * 3072
        mock_response.data = [mock_item]
        client.embeddings.create.return_value = mock_response
        return client

    @pytest.fixture
    def mock_collection(self):
        return MagicMock()

    def _make_query_results(self, distances, docs=None):
        """Build a ChromaDB-shaped query result dict."""
        n = len(distances)
        if docs is None:
            docs = [f"Doc {i}" for i in range(n)]
        return {
            "ids": [[f"id_{i}" for i in range(n)]],
            "metadatas": [[{"section": "Test", "chunk": i + 1} for i in range(n)]],
            "distances": [distances],
            "documents": [docs],
        }

    def test_chunks_below_threshold_are_retrieved(self, mock_oai_client, mock_collection):
        mock_collection.query.return_value = self._make_query_results(
            [0.3, 0.5],
            ["Close fact.", "Medium fact."],
        )
        context, n, sections = build_context_injection(
            mock_oai_client, mock_collection, "test query", d_threshold=0.825,
        )
        assert n == 2
        assert "Close fact." in context
        assert "Medium fact." in context

    def test_chunks_above_threshold_are_discarded(self, mock_oai_client, mock_collection):
        mock_collection.query.return_value = self._make_query_results(
            [0.3, 0.9],
            ["Close fact.", "Far fact."],
        )
        context, n, sections = build_context_injection(
            mock_oai_client, mock_collection, "test query", d_threshold=0.825,
        )
        assert n == 1
        assert "Close fact." in context
        assert "Far fact." not in context

    def test_no_chunks_below_threshold(self, mock_oai_client, mock_collection):
        mock_collection.query.return_value = self._make_query_results([0.9, 1.1])
        context, n, sections = build_context_injection(
            mock_oai_client, mock_collection, "test query", d_threshold=0.825,
        )
        assert n == 0
        assert sections == []
        assert "<retrieved_context></retrieved_context>" in context

    def test_sections_are_unique_and_ordered(self, mock_oai_client, mock_collection):
        results = self._make_query_results([0.2, 0.3, 0.4])
        results["metadatas"] = [[
            {"section": "Career", "chunk": 1},
            {"section": "Hobbies", "chunk": 1},
            {"section": "Career", "chunk": 2},
        ]]
        mock_collection.query.return_value = results
        _, _, sections = build_context_injection(
            mock_oai_client, mock_collection, "test query", d_threshold=0.825,
        )
        assert sections == ["Career", "Hobbies"]

    def test_embedding_failure_returns_graceful_fallback(self, mock_collection):
        client = MagicMock()
        client.embeddings.create.side_effect = Exception("API down")
        context, n, sections = build_context_injection(
            client, mock_collection, "test query",
        )
        assert n == 0
        assert "temporarily unavailable" in context

    def test_output_contains_xml_chunk_tags(self, mock_oai_client, mock_collection):
        mock_collection.query.return_value = self._make_query_results(
            [0.2], ["A biographical fact."],
        )
        context, n, _ = build_context_injection(
            mock_oai_client, mock_collection, "test query", d_threshold=0.825,
        )
        assert "<chunk source=" in context
        assert "<retrieved_context>" in context
        assert "A biographical fact." in context
