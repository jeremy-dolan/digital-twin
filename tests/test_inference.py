from unittest.mock import MagicMock

from inference import _normalize_mixed_history, _ThoughtAccordion


class TestNormalizeMixedHistory:

    def test_keeps_user_and_assistant_dicts(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _normalize_mixed_history(msgs)
        assert result == msgs

    def test_drops_developer_messages(self):
        msgs = [
            {"role": "developer", "content": "system prompt"},
            {"role": "user", "content": "hi"},
        ]
        result = _normalize_mixed_history(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_drops_dicts_without_content(self):
        msgs = [
            {"role": "user"},
            {"role": "user", "content": "real"},
        ]
        result = _normalize_mixed_history(msgs)
        assert len(result) == 1

    def test_extracts_text_from_response_output_message(self):
        msg = MagicMock()
        msg.role = "assistant"
        text_content = MagicMock()
        text_content.text = "model response"
        # Make isinstance check work: ResponseOutputText
        type(text_content).__name__ = "ResponseOutputText"
        msg.content = [text_content]

        # We need the isinstance checks to pass, so patch at the type level
        from openai.types.responses import ResponseOutputMessage, ResponseOutputText
        real_msg = MagicMock(spec=ResponseOutputMessage)
        real_msg.role = "assistant"
        real_text = MagicMock(spec=ResponseOutputText)
        real_text.text = "model response"
        real_msg.content = [real_text]

        result = _normalize_mixed_history([real_msg])
        assert len(result) == 1
        assert result[0] == {"role": "assistant", "content": "model response"}

    def test_empty_input(self):
        assert _normalize_mixed_history([]) == []

    def test_mixed_types_filters_correctly(self):
        msgs = [
            {"role": "developer", "content": "prompt"},
            {"role": "user", "content": "question"},
            {"type": "function_call_output", "call_id": "x", "output": "result"},
            {"role": "assistant", "content": "answer"},
        ]
        result = _normalize_mixed_history(msgs)
        assert len(result) == 2
        assert result[0]["content"] == "question"
        assert result[1]["content"] == "answer"


class TestThoughtAccordion:

    def test_initial_state(self):
        t = _ThoughtAccordion()
        assert t.chatmessage.content == ""
        assert t.chatmessage.metadata["status"] == "pending"
        assert not t.finalized

    def test_add_reasoning_summary_replaces(self):
        t = _ThoughtAccordion()
        t.add_reasoning_summary("r1", "First title")
        t.add_reasoning_summary("r1", "Revised title")
        content = t.chatmessage.content
        assert "Revised title" in content
        assert "First title" not in content

    def test_multiple_summaries_are_joined(self):
        t = _ThoughtAccordion()
        t.add_reasoning_summary("r1", "Part one")
        t.add_reasoning_summary("r2", "Part two")
        content = t.chatmessage.content
        assert "Part one" in content
        assert "Part two" in content

    def test_reasoning_summaries_and_tools_coexist(self):
        t = _ThoughtAccordion()
        t.add_reasoning_summary("r1", "Thinking about dice")
        t.set_tool_pending("item_1", "roll_dice")
        t.set_tool_result("item_1", "roll_dice", "4")
        content = t.chatmessage.content
        assert "Thinking about dice" in content
        assert "roll_dice: 4" in content

    def test_set_tool_pending(self):
        t = _ThoughtAccordion()
        t.set_tool_pending("item_1", "roll_dice")
        assert "roll_dice..." in t.chatmessage.content

    def test_set_tool_result_replaces_pending(self):
        t = _ThoughtAccordion()
        t.set_tool_pending("item_1", "roll_dice")
        t.set_tool_result("item_1", "roll_dice", "4")
        content = t.chatmessage.content
        assert "roll_dice: 4" in content
        assert "roll_dice..." not in content

    def test_finalize_with_content(self):
        t = _ThoughtAccordion()
        t.add_reasoning_summary("r1", "Pondering")
        t.finalize()
        assert t.finalized
        assert "status" not in t.chatmessage.metadata
        assert "duration" in t.chatmessage.metadata
        assert "..." not in t.chatmessage.metadata["title"]
        assert t.chatmessage.content == "Pondering"

    def test_finalize_empty_summary(self):
        t = _ThoughtAccordion()
        t.finalize()
        assert t.finalized
        assert t.chatmessage.metadata["status"] == "done"
        assert "duration" in t.chatmessage.metadata
        assert "..." not in t.chatmessage.metadata["title"]
        assert t.chatmessage.content  # flavor text was added

    def test_finalize_is_idempotent(self):
        t = _ThoughtAccordion()
        t.add_reasoning_summary("r1", "Pondering")
        t.finalize()
        duration = t.chatmessage.metadata["duration"]
        t.finalize()
        assert t.chatmessage.metadata["duration"] == duration
