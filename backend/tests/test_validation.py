# tests/test_validation.py

import pytest


# ---------------------------------------------------------------------------
# SumRule tests
# ---------------------------------------------------------------------------

def test_sum_rule_pass():
    """SumRule returns PASS when line item amounts sum to the total."""
    from validation.rules.arithmetic import SumRule

    rule = SumRule(item_field="line_items", total_field="subtotal")
    fields = {
        "line_items": [
            {"description": "Widget A", "amount": "1000.00"},
            {"description": "Widget B", "amount": "500.00"},
            {"description": "Widget C", "amount": "250.00"},
        ],
        "subtotal": "1750.00",
    }
    results = rule.validate(fields)

    assert len(results) == 1
    assert results[0].status == "PASS"
    assert results[0].rule_code == "VAL_101"


def test_sum_rule_fail():
    """SumRule returns FAIL when line item amounts do not match total."""
    from validation.rules.arithmetic import SumRule

    rule = SumRule(item_field="line_items", total_field="subtotal", blocking=True)
    fields = {
        "line_items": [
            {"amount": "1000.00"},
            {"amount": "500.00"},
        ],
        "subtotal": "2000.00",  # wrong — should be 1500
    }
    results = rule.validate(fields)

    assert len(results) == 1
    assert results[0].status == "FAIL"
    assert results[0].blocking is True
    assert "1500" in results[0].message
    assert "2000" in results[0].message


def test_sum_rule_skips_when_fields_absent():
    """SumRule returns [] when required fields are not present."""
    from validation.rules.arithmetic import SumRule

    rule = SumRule(item_field="line_items", total_field="subtotal")

    # Neither field present
    assert rule.validate({}) == []

    # Only one field present
    assert rule.validate({"subtotal": "1000"}) == []
    assert rule.validate({"line_items": [{"amount": "500"}]}) == []


def test_sum_rule_handles_currency_symbols():
    """SumRule correctly strips currency symbols before parsing."""
    from validation.rules.arithmetic import SumRule

    rule = SumRule(item_field="line_items", total_field="subtotal")
    fields = {
        "line_items": [
            {"amount": "₹1,000.00"},
            {"amount": "₹500.00"},
        ],
        "subtotal": "₹1,500.00",
    }
    results = rule.validate(fields)
    assert results[0].status == "PASS"


# ---------------------------------------------------------------------------
# TotalConsistencyRule tests
# ---------------------------------------------------------------------------

def test_total_consistency_pass():
    """TotalConsistencyRule passes when subtotal + tax == total."""
    from validation.rules.arithmetic import TotalConsistencyRule

    rule = TotalConsistencyRule()
    fields = {
        "subtotal":     "1000.00",
        "tax_amount":   "180.00",
        "total_amount": "1180.00",
    }
    results = rule.validate(fields)

    assert len(results) == 1
    assert results[0].status == "PASS"
    assert results[0].rule_code == "VAL_102"


def test_total_consistency_fail():
    """TotalConsistencyRule fails when subtotal + tax != total."""
    from validation.rules.arithmetic import TotalConsistencyRule

    rule = TotalConsistencyRule()
    fields = {
        "subtotal":     "1000.00",
        "tax_amount":   "180.00",
        "total_amount": "1500.00",  # wrong
    }
    results = rule.validate(fields)

    assert len(results) == 1
    assert results[0].status == "FAIL"
    assert results[0].blocking is True
    assert results[0].severity == "ERROR"


def test_total_consistency_skips_when_fields_absent():
    """TotalConsistencyRule skips when any of the three fields is missing."""
    from validation.rules.arithmetic import TotalConsistencyRule

    rule = TotalConsistencyRule()
    assert rule.validate({"subtotal": "1000", "tax_amount": "100"}) == []
    assert rule.validate({"subtotal": "1000", "total_amount": "1100"}) == []
    assert rule.validate({}) == []


# ---------------------------------------------------------------------------
# DateOrderRule tests
# ---------------------------------------------------------------------------

def test_date_order_pass():
    """DateOrderRule passes when start date is before end date."""
    from validation.rules.logic import DateOrderRule

    rule = DateOrderRule(start_field="invoice_date", end_field="due_date")
    fields = {
        "invoice_date": "2024-01-01",
        "due_date":     "2024-02-01",
    }
    results = rule.validate(fields)

    assert len(results) == 1
    assert results[0].status == "PASS"
    assert results[0].rule_code == "VAL_201"


def test_date_order_fail():
    """DateOrderRule fails when start date is after end date."""
    from validation.rules.logic import DateOrderRule

    rule = DateOrderRule(
        start_field="effective_date",
        end_field="expiry_date",
        blocking=True,
    )
    fields = {
        "effective_date": "2025-06-01",
        "expiry_date":    "2024-01-01",  # before effective — invalid
    }
    results = rule.validate(fields)

    assert len(results) == 1
    assert results[0].status == "FAIL"
    assert results[0].blocking is True


def test_date_order_skips_when_fields_absent():
    """DateOrderRule returns [] when either date is missing."""
    from validation.rules.logic import DateOrderRule

    rule = DateOrderRule(start_field="invoice_date", end_field="due_date")
    assert rule.validate({}) == []
    assert rule.validate({"invoice_date": "2024-01-01"}) == []


def test_date_order_warns_on_unparseable_dates():
    """DateOrderRule returns WARNING when dates cannot be parsed."""
    from validation.rules.logic import DateOrderRule

    rule = DateOrderRule(start_field="start", end_field="end")
    fields = {"start": "not-a-date", "end": "also-not-a-date"}
    results = rule.validate(fields)

    assert len(results) == 1
    assert results[0].status == "WARNING"
    assert results[0].blocking is False


def test_date_order_handles_multiple_formats():
    """DateOrderRule parses common Indian and international date formats."""
    from validation.rules.logic import DateOrderRule

    rule = DateOrderRule(start_field="start", end_field="end")

    format_pairs = [
        ("01/01/2024", "01/02/2024"),   # DD/MM/YYYY
        ("2024-01-01", "2024-02-01"),   # ISO
        ("01-01-2024", "01-02-2024"),   # DD-MM-YYYY
        ("January 1, 2024", "February 1, 2024"),
    ]
    for start, end in format_pairs:
        results = rule.validate({"start": start, "end": end})
        assert results[0].status == "PASS", \
            f"Failed to parse date pair: {start}, {end}"


# ---------------------------------------------------------------------------
# ConditionalRequiredRule tests
# ---------------------------------------------------------------------------

def test_conditional_required_triggers():
    """ConditionalRequiredRule fails when condition is met and field is missing."""
    from validation.rules.logic import ConditionalRequiredRule

    rule = ConditionalRequiredRule(
        condition_field="employment_type",
        condition_value="salaried",
        required_field="employer_name",
        blocking=False,
    )
    fields = {"employment_type": "salaried"}  # employer_name missing

    results = rule.validate(fields)

    assert len(results) == 1
    assert results[0].status == "FAIL"
    assert results[0].field == "employer_name"


def test_conditional_required_passes_when_field_present():
    """ConditionalRequiredRule passes when condition met and required field present."""
    from validation.rules.logic import ConditionalRequiredRule

    rule = ConditionalRequiredRule(
        condition_field="employment_type",
        condition_value="salaried",
        required_field="employer_name",
    )
    fields = {
        "employment_type": "salaried",
        "employer_name":   "Acme Corp",
    }
    results = rule.validate(fields)

    assert results[0].status == "PASS"


def test_conditional_required_skips_when_not_met():
    """ConditionalRequiredRule returns [] when condition is not met."""
    from validation.rules.logic import ConditionalRequiredRule

    rule = ConditionalRequiredRule(
        condition_field="employment_type",
        condition_value="salaried",
        required_field="employer_name",
    )
    # condition not met — self-employed doesn't need employer_name
    fields = {"employment_type": "self-employed"}

    results = rule.validate(fields)
    assert results == [], "Rule should not apply when condition is not met"


def test_conditional_required_case_insensitive():
    """ConditionalRequiredRule comparison is case-insensitive."""
    from validation.rules.logic import ConditionalRequiredRule

    rule = ConditionalRequiredRule(
        condition_field="employment_type",
        condition_value="salaried",
        required_field="employer_name",
    )
    # "Salaried" vs "salaried" — should still trigger
    fields = {"employment_type": "Salaried"}

    results = rule.validate(fields)
    assert results[0].status == "FAIL"


# ---------------------------------------------------------------------------
# RequiredFieldsRule tests
# ---------------------------------------------------------------------------

def test_required_fields_all_present():
    """RequiredFieldsRule passes for all fields when all present."""
    from validation.rules.completeness import RequiredFieldsRule

    rule = RequiredFieldsRule(
        required_fields=["vendor_name", "invoice_number", "total_amount"]
    )
    fields = {
        "vendor_name":    "Acme Ltd",
        "invoice_number": "INV-001",
        "total_amount":   "5000.00",
    }
    results = rule.validate(fields)

    assert all(r.status == "PASS" for r in results)
    assert len(results) == 3


def test_required_fields_some_missing():
    """RequiredFieldsRule returns FAIL for each missing field."""
    from validation.rules.completeness import RequiredFieldsRule

    rule = RequiredFieldsRule(
        required_fields=["vendor_name", "invoice_number", "total_amount"],
        blocking=True,
    )
    fields = {"vendor_name": "Acme Ltd"}  # invoice_number and total_amount missing

    results = rule.validate(fields)
    failed = [r for r in results if r.status == "FAIL"]

    assert len(failed) == 2
    failed_fields = {r.field for r in failed}
    assert "invoice_number" in failed_fields
    assert "total_amount" in failed_fields
    assert all(r.blocking for r in failed)


# ---------------------------------------------------------------------------
# Ruleset loading tests
# ---------------------------------------------------------------------------

def test_invoice_ruleset_loads():
    """Invoice ruleset loads without error and returns a non-empty rule list."""
    from validation.rulesets.invoice import get_rules

    rules = get_rules()
    assert len(rules) > 0, "Invoice ruleset must have at least one rule"

    # All rules must have get_code() and get_name()
    for rule in rules:
        assert rule.get_code().startswith("VAL_"), \
            f"Rule code must start with VAL_: {rule.get_code()}"
        assert len(rule.get_name()) > 0


def test_all_rulesets_load():
    """All 6 rulesets load without import errors."""
    from validation.rulesets.invoice import get_rules as inv
    from validation.rulesets.cv_resume import get_rules as cv
    from validation.rulesets.gst_return import get_rules as gst
    from validation.rulesets.contract import get_rules as con
    from validation.rulesets.bank_statement import get_rules as bank
    from validation.rulesets.loan_application import get_rules as loan

    for name, fn in [
        ("invoice", inv), ("cv_resume", cv), ("gst_return", gst),
        ("contract", con), ("bank_statement", bank), ("loan_application", loan),
    ]:
        rules = fn()
        assert isinstance(rules, list), f"{name} ruleset must return a list"
        assert len(rules) > 0, f"{name} ruleset must not be empty"


# ---------------------------------------------------------------------------
# ValidationEngine tests
# ---------------------------------------------------------------------------

def test_validation_engine_returns_contract_shape():
    """
    ValidationEngine.validate() returns a dict matching Contract 4 exactly.
    All required keys present, correct types.
    """
    from validation.engine import ValidationEngine

    engine = ValidationEngine()
    result = engine.validate(
        extracted_fields={
            "vendor_name":    "Acme Ltd",
            "invoice_number": "INV-001",
            "invoice_date":   "2024-01-01",
            "due_date":       "2024-02-01",
            "subtotal":       "1000.00",
            "tax_amount":     "180.00",
            "total_amount":   "1180.00",
            "line_items":     [{"amount": "1000.00"}],
        },
        doc_type="invoice",
    )

    # Required top-level keys
    for key in ["doc_type", "rules_run", "passed", "failed", "warnings",
                "blocking_failures", "is_valid", "results"]:
        assert key in result, f"Missing key in validation result: {key}"

    assert result["doc_type"] == "invoice"
    assert isinstance(result["rules_run"], int)
    assert isinstance(result["is_valid"], bool)
    assert isinstance(result["results"], list)

    # Each result item must match Contract 4 shape
    for r in result["results"]:
        for field in ["field", "rule", "rule_code", "status",
                      "expected", "actual", "message", "severity", "blocking"]:
            assert field in r, f"Missing field in result item: {field}"
        assert r["status"] in ("PASS", "FAIL", "WARNING", "SKIPPED")
        assert r["severity"] in ("ERROR", "WARNING", "INFO")
        assert isinstance(r["blocking"], bool)


def test_validation_engine_unknown_doc_type_returns_empty():
    """ValidationEngine returns empty valid result for unknown doc types."""
    from validation.engine import ValidationEngine

    engine = ValidationEngine()
    result = engine.validate({}, doc_type="unknown_type_xyz")

    assert result["rules_run"] == 0
    assert result["is_valid"] is True
    assert result["results"] == []


def test_validation_engine_is_valid_false_on_blocking_failure():
    """is_valid is False when there is at least one blocking failure."""
    from validation.engine import ValidationEngine

    engine = ValidationEngine()
    result = engine.validate(
        extracted_fields={
            # Missing required fields — will trigger blocking RequiredFieldsRule
            "subtotal": "1000",
            "tax_amount": "100",
            "total_amount": "9999",  # wrong total — TotalConsistencyRule will FAIL
        },
        doc_type="invoice",
    )

    assert result["is_valid"] is False
    assert result["blocking_failures"] > 0


def test_validation_engine_rule_failure_doesnt_crash_engine():
    """A rule that raises internally should be skipped, not crash the engine."""
    from validation.engine import ValidationEngine
    from validation.rules.base import BaseRule, ValidationResult

    class BrokenRule(BaseRule):
        def get_code(self): return "VAL_999"
        def get_name(self): return "broken_rule"
        def validate(self, fields): raise RuntimeError("Simulated rule crash")

    engine = ValidationEngine()

    # Manually inject broken rule
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(
            "validation.engine.ValidationEngine._load_rules",
            lambda self, dt: [BrokenRule()]
        )
        result = engine.validate({}, doc_type="invoice")

    # Engine should still return a valid result dict
    assert "is_valid" in result
    assert result["rules_run"] == 1
    assert result["passed"] == 0