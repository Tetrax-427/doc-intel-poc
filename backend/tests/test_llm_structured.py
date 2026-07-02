"""
tests/test_llm_structured.py

Unit tests for llm/structured.py:
    - build_extraction_model
    - call_structured (Instructor wrapper)
    - All Pydantic response models (shape + defaults)

Changes from previous version:
- call_structured() now takes system: str, user: str instead of
  messages: list[dict]. The Anthropic system extraction test has been
  updated to reflect this — there is no longer any extraction from a
  messages list; system is passed directly.
- call_structured() returns (result, usage) tuple, not bare result.
  Tests now unpack accordingly.
- log_usage patching removed — llm.usage no longer exists.
  Instructor's internal machinery is still mocked via the instructor module.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# build_extraction_model
# ---------------------------------------------------------------------------

class TestBuildExtractionModel:

    def test_creates_model_with_correct_fields(self):
        from llm.structured import build_extraction_model
        Model = build_extraction_model({
            "vendor_name":  "Name of the vendor",
            "total_amount": "Total invoice amount",
        })
        instance = Model(vendor_name="Acme", total_amount="$100")
        assert instance.vendor_name == "Acme"
        assert instance.total_amount == "$100"

    def test_all_fields_optional_with_none_default(self):
        from llm.structured import build_extraction_model
        Model = build_extraction_model({
            "invoice_date": "Date of invoice",
            "due_date":     "Payment due date",
        })
        instance = Model()
        assert instance.invoice_date is None
        assert instance.due_date is None

    def test_model_named_extraction_result(self):
        from llm.structured import build_extraction_model
        Model = build_extraction_model({"field": "desc"})
        assert Model.__name__ == "ExtractionResult"

    def test_sanitises_field_names_with_spaces(self):
        from llm.structured import build_extraction_model
        Model = build_extraction_model({"vendor name": "Vendor name"})
        instance = Model(**{"vendor_name": "Acme"})
        assert instance.vendor_name == "Acme"

    def test_raises_on_empty_fields(self):
        from llm.structured import build_extraction_model
        with pytest.raises(ValueError):
            build_extraction_model({})

    def test_is_pydantic_base_model(self):
        from llm.structured import build_extraction_model
        Model = build_extraction_model({"key": "desc"})
        assert issubclass(Model, BaseModel)

    def test_model_dump_returns_dict(self):
        from llm.structured import build_extraction_model
        Model = build_extraction_model({"amount": "Invoice amount"})
        instance = Model(amount="$500")
        assert instance.model_dump()["amount"] == "$500"


# ---------------------------------------------------------------------------
# Response model shapes
# ---------------------------------------------------------------------------

class TestDocumentClassification:

    def test_required_fields(self):
        from llm.structured import DocumentClassification
        obj = DocumentClassification(
            doc_type="invoice",
            confidence=0.95,
            reasoning="Has line items and total",
            key_signals=["Invoice #", "Total Due"],
        )
        assert obj.doc_type == "invoice"
        assert obj.confidence == 0.95
        assert len(obj.key_signals) == 2

    def test_key_signals_defaults_to_empty_list(self):
        from llm.structured import DocumentClassification
        obj = DocumentClassification(
            doc_type="general", confidence=0.5, reasoning="unclear"
        )
        assert obj.key_signals == []


class TestQueryExpansion:

    def test_holds_expanded_query(self):
        from llm.structured import QueryExpansion
        obj = QueryExpansion(expanded_query="What is the total invoice amount?")
        assert "invoice" in obj.expanded_query


class TestDocumentSummary:

    def test_all_list_fields_default_to_empty(self):
        from llm.structured import DocumentSummary
        obj = DocumentSummary(
            short="A contract",
            overview="Two-party agreement",
            document_type="Contract",
        )
        assert obj.key_topics == []
        assert obj.entities == []
        assert obj.dates == []
        assert obj.amounts == []

    def test_full_construction(self):
        from llm.structured import DocumentSummary
        obj = DocumentSummary(
            short="Invoice from Acme",
            overview="Monthly service invoice",
            key_topics=["services", "billing"],
            entities=["Acme Corp"],
            dates=["2024-01-01"],
            amounts=["$5000"],
            document_type="Invoice",
        )
        assert obj.document_type == "Invoice"
        assert obj.amounts == ["$5000"]


class TestTableModels:

    def test_table_item_defaults(self):
        from llm.structured import TableItem
        item = TableItem(title="Revenue Table")
        assert item.headers == []
        assert item.rows == []
        assert item.chart_type == "bar"

    def test_table_list_defaults(self):
        from llm.structured import TableList
        lst = TableList()
        assert lst.tables == []

    def test_table_list_with_items(self):
        from llm.structured import TableList, TableItem
        lst = TableList(tables=[
            TableItem(title="T1", headers=["A", "B"], rows=[["1", "2"]], chart_type="bar"),
            TableItem(title="T2", headers=["X"], rows=[], chart_type="line"),
        ])
        assert len(lst.tables) == 2
        assert lst.tables[0].title == "T1"

    def test_model_dump_produces_list_of_dicts(self):
        from llm.structured import TableList, TableItem
        lst = TableList(tables=[TableItem(title="Sales")])
        dumped = [t.model_dump() for t in lst.tables]
        assert isinstance(dumped[0], dict)
        assert dumped[0]["title"] == "Sales"


class TestSchemaResult:

    def test_extra_fields_allowed(self):
        from llm.structured import SchemaResult
        obj = SchemaResult(**{
            "vendor_name":  "Name of the vendor",
            "total_amount": "Total amount due",
        })
        d = obj.model_dump()
        assert d["vendor_name"] == "Name of the vendor"
        assert d["total_amount"] == "Total amount due"

    def test_empty_schema_result(self):
        from llm.structured import SchemaResult
        obj = SchemaResult()
        assert obj.model_dump() == {}


# ---------------------------------------------------------------------------
# call_structured
# ---------------------------------------------------------------------------

class TestCallStructured:

    @patch("llm.structured.instructor")
    def test_returns_tuple_of_result_and_usage_for_openai(self, mock_instructor):
        """call_structured() returns (BaseModel, usage_dict | None) — not bare model."""
        from llm.structured import call_structured, DocumentClassification

        expected = DocumentClassification(
            doc_type="invoice", confidence=0.9,
            reasoning="has line items", key_signals=["Invoice #"],
        )
        # Simulate instructor returning a model with _raw_response.usage attached
        usage_mock = MagicMock()
        usage_mock.prompt_tokens = 120
        usage_mock.completion_tokens = 15
        usage_mock.total_tokens = 135
        expected._raw_response = MagicMock(usage=usage_mock)

        mock_ic = MagicMock()
        mock_ic.chat.completions.create.return_value = expected
        mock_instructor.from_openai.return_value = mock_ic

        result, usage = call_structured(
            raw_client=MagicMock(),
            provider="openai",
            model="gpt-4o-mini",
            system="Classify the document.",
            user="Document text here.",
            response_model=DocumentClassification,
        )
        assert isinstance(result, DocumentClassification)
        assert result.doc_type == "invoice"
        assert usage is not None
        assert usage["prompt_tokens"] == 120
        assert usage["completion_tokens"] == 15
        assert usage["total_tokens"] == 135

    @patch("llm.structured.instructor")
    def test_returns_tuple_of_result_and_usage_for_anthropic(self, mock_instructor):
        from llm.structured import call_structured, DocumentClassification

        expected = DocumentClassification(
            doc_type="resume", confidence=0.85,
            reasoning="has work history", key_signals=["Experience"],
        )
        usage_mock = MagicMock()
        usage_mock.input_tokens = 80
        usage_mock.output_tokens = 12
        # Anthropic does not have total_tokens — usage extraction computes it
        del usage_mock.total_tokens  # ensure it's absent
        expected._raw_response = MagicMock(usage=usage_mock)

        mock_ic = MagicMock()
        mock_ic.messages.create.return_value = expected
        mock_instructor.from_anthropic.return_value = mock_ic

        result, usage = call_structured(
            raw_client=MagicMock(),
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
            system="Classify the document.",
            user="Document text here.",
            response_model=DocumentClassification,
        )
        assert result.doc_type == "resume"
        assert usage is not None
        assert usage["prompt_tokens"] == 80
        assert usage["completion_tokens"] == 12
        assert usage["total_tokens"] == 92  # computed as input + output
        mock_ic.messages.create.assert_called_once()

    @patch("llm.structured.instructor")
    def test_usage_is_none_when_raw_response_absent(self, mock_instructor):
        """If instructor version doesn't attach _raw_response, usage degrades to None."""
        from llm.structured import call_structured, QueryExpansion

        expected = QueryExpansion(expanded_query="expanded query text")
        # No _raw_response attached — simulates older instructor versions
        if hasattr(expected, "_raw_response"):
            del expected._raw_response

        mock_ic = MagicMock()
        mock_ic.chat.completions.create.return_value = expected
        mock_instructor.from_openai.return_value = mock_ic

        result, usage = call_structured(
            raw_client=MagicMock(),
            provider="openai",
            model="gpt-4o-mini",
            system="Expand the query.",
            user="What is the invoice total?",
            response_model=QueryExpansion,
        )
        assert result.expanded_query == "expanded query text"
        assert usage is None  # graceful degradation, not an error

    @patch("llm.structured.instructor")
    def test_raises_structured_output_error_on_failure(self, mock_instructor):
        from llm.structured import call_structured, DocumentClassification
        from core.errors import StructuredOutputError

        mock_ic = MagicMock()
        mock_ic.chat.completions.create.side_effect = Exception("validation failed")
        mock_instructor.from_openai.return_value = mock_ic

        with pytest.raises(StructuredOutputError) as exc_info:
            call_structured(
                raw_client=MagicMock(),
                provider="openai",
                model="gpt-4o-mini",
                system="Classify.",
                user="Document text.",
                response_model=DocumentClassification,
            )
        assert exc_info.value.context["response_model"] == "DocumentClassification"
        assert exc_info.value.context["provider"] == "openai"

    @patch("llm.structured.instructor")
    def test_raises_llm_config_error_for_unsupported_provider(self, mock_instructor):
        from llm.structured import call_structured, DocumentClassification
        from core.errors import LLMConfigError

        with pytest.raises(LLMConfigError):
            call_structured(
                raw_client=MagicMock(),
                provider="cohere",
                model="command-r",
                system="Classify.",
                user="Document text.",
                response_model=DocumentClassification,
            )

    @patch("llm.structured.instructor")
    def test_anthropic_receives_system_as_top_level_kwarg(self, mock_instructor):
        """
        For Anthropic, system must be a top-level kwarg to messages.create(),
        NOT embedded inside the messages list. This is the new design —
        system/user are passed explicitly, there is no extraction from a
        messages list.
        """
        from llm.structured import call_structured, QueryExpansion

        expected = QueryExpansion(expanded_query="expanded")
        expected._raw_response = MagicMock(
            usage=MagicMock(input_tokens=10, output_tokens=5)
        )

        mock_ic = MagicMock()
        mock_ic.messages.create.return_value = expected
        mock_instructor.from_anthropic.return_value = mock_ic

        call_structured(
            raw_client=MagicMock(),
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
            system="You are a query expander.",
            user="Expand this question.",
            response_model=QueryExpansion,
        )

        call_kwargs = mock_ic.messages.create.call_args.kwargs
        # system is a top-level kwarg (may be list for cache_control or plain string)
        assert "system" in call_kwargs
        # user content in messages list, system role must NOT appear there
        messages = call_kwargs.get("messages", [])
        assert any(m["role"] == "user" for m in messages)
        assert all(m["role"] != "system" for m in messages)

    @patch("llm.structured.instructor")
    def test_openai_sends_system_role_message(self, mock_instructor):
        """For OpenAI/Groq, system appears as a role='system' message."""
        from llm.structured import call_structured, QueryExpansion

        expected = QueryExpansion(expanded_query="expanded")
        expected._raw_response = MagicMock(
            usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        )

        mock_ic = MagicMock()
        mock_ic.chat.completions.create.return_value = expected
        mock_instructor.from_openai.return_value = mock_ic

        call_structured(
            raw_client=MagicMock(),
            provider="openai",
            model="gpt-4o-mini",
            system="You are a query expander.",
            user="Expand this question.",
            response_model=QueryExpansion,
        )

        call_kwargs = mock_ic.chat.completions.create.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles
        system_content = next(m["content"] for m in messages if m["role"] == "system")
        assert system_content == "You are a query expander."