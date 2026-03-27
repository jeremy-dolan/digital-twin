"""Tests for rag.py: indexing pipeline (embed_chunks, db_store/load) and
query-time functions (format_injection, retrieve_context)."""

from unittest.mock import MagicMock

import chromadb
import pytest

from rag import (
    ChunkedText,
    db_load_embeds,
    db_store_embeds,
    embed_chunks,
    format_injection,
    retrieve_context,
)


class TestEmbedChunks:

    @pytest.fixture
    def mock_oai_client(self):
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2]),
            MagicMock(embedding=[0.3, 0.4]),
        ]
        client.embeddings.create.return_value = mock_response
        return client

    def test_attaches_embeddings_to_chunks(self, mock_oai_client):
        chunks = [
            ChunkedText(text="Fact one.", metadata={"section": "A", "chunk": 1}),
            ChunkedText(text="Fact two.", metadata={"section": "A", "chunk": 2}),
        ]
        result = embed_chunks(mock_oai_client, chunks)
        assert result is chunks  # mutates in place
        assert chunks[0].embedding == [0.1, 0.2]
        assert chunks[1].embedding == [0.3, 0.4]

    def test_adds_embedding_model_to_metadata(self, mock_oai_client):
        chunks = [ChunkedText(text="Fact.", metadata={"section": "A", "chunk": 1})]
        mock_oai_client.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.0])
        ]
        embed_chunks(mock_oai_client, chunks)
        assert "embedding_model" in chunks[0].metadata

    def test_passes_chunk_texts_to_api(self, mock_oai_client):
        chunks = [
            ChunkedText(text="Alpha.", metadata={"section": "A", "chunk": 1}),
            ChunkedText(text="Beta.", metadata={"section": "A", "chunk": 2}),
        ]
        embed_chunks(mock_oai_client, chunks)
        call_args = mock_oai_client.embeddings.create.call_args
        assert call_args.kwargs["input"] == ["Alpha.", "Beta."]


class TestDbRoundTrip:
    """Store chunks into ChromaDB and load them back, verifying nothing is lost."""

    @pytest.fixture
    def chroma_client(self):
        return chromadb.EphemeralClient()

    COLLECTION = "test_roundtrip"

    def _make_chunks(self):
        return [
            ChunkedText(
                text="Born in Los Angeles.",
                metadata={"section": "Personal", "chunk": 1, "guidance": "Only if asked"},
                embedding=[0.1] * 32,
            ),
            ChunkedText(
                text="Works as a software engineer.",
                metadata={"section": "Career", "chunk": 1},
                embedding=[0.2] * 32,
            ),
        ]

    def test_roundtrip_preserves_texts(self, chroma_client):
        chunks = self._make_chunks()
        db_store_embeds(chroma_client, self.COLLECTION, chunks)
        loaded = db_load_embeds(chroma_client, self.COLLECTION)
        loaded_texts = {c.text for c in loaded}
        assert loaded_texts == {"Born in Los Angeles.", "Works as a software engineer."}

    def test_roundtrip_preserves_metadata(self, chroma_client):
        chunks = self._make_chunks()
        db_store_embeds(chroma_client, self.COLLECTION, chunks)
        loaded = db_load_embeds(chroma_client, self.COLLECTION)
        by_text = {c.text: c for c in loaded}
        assert by_text["Born in Los Angeles."].metadata["section"] == "Personal"
        assert by_text["Born in Los Angeles."].metadata["guidance"] == "Only if asked"
        assert by_text["Works as a software engineer."].metadata["section"] == "Career"

    def test_roundtrip_preserves_embeddings(self, chroma_client):
        chunks = self._make_chunks()
        db_store_embeds(chroma_client, self.COLLECTION, chunks)
        loaded = db_load_embeds(chroma_client, self.COLLECTION)
        by_text = {c.text: c for c in loaded}
        assert by_text["Born in Los Angeles."].embedding[:3] == pytest.approx([0.1] * 3)

    def test_store_replaces_existing_collection(self, chroma_client):
        chunks_v1 = [ChunkedText(text="Old.", metadata={"section": "S", "chunk": 1}, embedding=[0.5] * 32)]
        chunks_v2 = [ChunkedText(text="New.", metadata={"section": "S", "chunk": 1}, embedding=[0.6] * 32)]
        db_store_embeds(chroma_client, self.COLLECTION, chunks_v1)
        db_store_embeds(chroma_client, self.COLLECTION, chunks_v2)
        loaded = db_load_embeds(chroma_client, self.COLLECTION)
        assert len(loaded) == 1
        assert loaded[0].text == "New."

    def test_loaded_chunks_have_ids(self, chroma_client):
        chunks = self._make_chunks()
        db_store_embeds(chroma_client, self.COLLECTION, chunks)
        loaded = db_load_embeds(chroma_client, self.COLLECTION)
        assert all(c.id is not None for c in loaded)


class TestFormatInjection:

    def test_retrieval_failure(self):
        result = format_injection(retrieval_failure=True)
        assert "temporarily unavailable" in result
        assert "<retrieved_context></retrieved_context>" in result

    def test_no_chunks(self):
        result = format_injection(retrieved_chunks=None)
        assert "No relevant biographical excerpts" in result
        assert "<retrieved_context></retrieved_context>" in result

    def test_empty_list(self):
        result = format_injection(retrieved_chunks=[])
        assert "No relevant biographical excerpts" in result

    def test_chunks_without_guidance(self):
        chunks = [
            {"id": "abc", "metadata": {"section": "Career", "chunk": 1}, "document": "He works at Acme."},
        ]
        result = format_injection(chunks)
        assert '<chunk source="abc">' in result
        assert "He works at Acme." in result
        assert "<guidance>" not in result
        assert "may be relevant" in result

    def test_chunks_with_guidance(self):
        chunks = [
            {
                "id": "xyz",
                "metadata": {"section": "Personal", "chunk": 1, "guidance": "Only mention if asked"},
                "document": "Born in LA.",
            },
        ]
        result = format_injection(chunks)
        assert '<chunk source="xyz">' in result
        assert "Born in LA." in result
        assert "<guidance>Only mention if asked</guidance>" in result

    def test_multiple_chunks(self):
        chunks = [
            {"id": "a", "metadata": {"section": "S", "chunk": 1}, "document": "Fact A."},
            {"id": "b", "metadata": {"section": "S", "chunk": 2, "guidance": "Be careful"}, "document": "Fact B."},
        ]
        result = format_injection(chunks)
        assert '<chunk source="a">' in result
        assert '<chunk source="b">' in result
        assert "Fact A." in result
        assert "Fact B." in result
        assert "<guidance>Be careful</guidance>" in result


class TestRetrieveContext:
    """Test the retrieval filtering and formatting logic by mocking the OpenAI
    embedding call and ChromaDB collection.query() return value."""

    @pytest.fixture
    def mock_oai_client(self):
        client = MagicMock()
        mock_response = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.0] * 3072
        mock_response.data = [mock_item]
        client.embeddings.create.return_value = mock_response
        return client

    @pytest.fixture
    def mock_collection(self):
        return MagicMock()

    def _make_query_results(self, distances, docs=None, metadatas=None):
        """Build a ChromaDB-shaped query result dict."""
        n = len(distances)
        if docs is None:
            docs = [f"Doc {i}" for i in range(n)]
        if metadatas is None:
            metadatas = [{"section": "Test", "chunk": i + 1} for i in range(n)]
        return {
            "ids": [[f"id_{i}" for i in range(n)]],
            "metadatas": [metadatas],
            "distances": [distances],
            "documents": [docs],
        }

    def test_chunks_below_threshold_are_retrieved(self, mock_oai_client, mock_collection):
        mock_collection.query.return_value = self._make_query_results(
            [0.3, 0.5],
            ["Close fact.", "Medium fact."],
        )
        context, n, sections = retrieve_context(
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
        context, n, sections = retrieve_context(
            mock_oai_client, mock_collection, "test query", d_threshold=0.825,
        )
        assert n == 1
        assert "Close fact." in context
        assert "Far fact." not in context

    def test_no_chunks_below_threshold(self, mock_oai_client, mock_collection):
        mock_collection.query.return_value = self._make_query_results([0.9, 1.1])
        context, n, sections = retrieve_context(
            mock_oai_client, mock_collection, "test query", d_threshold=0.825,
        )
        assert n == 0
        assert sections == []
        assert "<retrieved_context></retrieved_context>" in context

    def test_sections_are_unique_and_ordered(self, mock_oai_client, mock_collection):
        results = self._make_query_results(
            [0.2, 0.3, 0.4],
            metadatas=[
                {"section": "Career", "chunk": 1},
                {"section": "Hobbies", "chunk": 1},
                {"section": "Career", "chunk": 2},
            ],
        )
        mock_collection.query.return_value = results
        _, _, sections = retrieve_context(
            mock_oai_client, mock_collection, "test query", d_threshold=0.825,
        )
        assert sections == ["Career", "Hobbies"]

    def test_embedding_failure_returns_graceful_fallback(self, mock_collection):
        client = MagicMock()
        client.embeddings.create.side_effect = Exception("API down")
        context, n, sections = retrieve_context(
            client, mock_collection, "test query",
        )
        assert n == 0
        assert "temporarily unavailable" in context

    def test_collection_query_failure_returns_graceful_fallback(self, mock_oai_client, mock_collection):
        mock_collection.query.side_effect = Exception("ChromaDB error")
        context, n, sections = retrieve_context(
            mock_oai_client, mock_collection, "test query",
        )
        assert n == 0
        assert "temporarily unavailable" in context

    def test_output_contains_xml_chunk_tags(self, mock_oai_client, mock_collection):
        mock_collection.query.return_value = self._make_query_results(
            [0.2], ["A biographical fact."],
        )
        context, n, _ = retrieve_context(
            mock_oai_client, mock_collection, "test query", d_threshold=0.825,
        )
        assert "<chunk source=" in context
        assert "<retrieved_context>" in context
        assert "A biographical fact." in context

    def test_guidance_metadata_rendered_in_output(self, mock_oai_client, mock_collection):
        mock_collection.query.return_value = self._make_query_results(
            [0.2],
            ["Born in LA."],
            metadatas=[{"section": "Personal", "chunk": 1, "guidance": "Only if asked directly"}],
        )
        context, n, _ = retrieve_context(
            mock_oai_client, mock_collection, "test query", d_threshold=0.825,
        )
        assert n == 1
        assert "Born in LA." in context
        assert "<guidance>Only if asked directly</guidance>" in context
