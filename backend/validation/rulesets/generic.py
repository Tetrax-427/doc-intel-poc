"""
validation/rulesets/generic.py — Schema-agnostic rules for dynamically
generated (user-defined) extraction schemas, where field names aren't known
in advance so the doc-type-specific rulesets (invoice.py, cv_resume.py, etc.)
can't apply.

Wired in via validation/engine.py's RULESET_MAP under "dynamic" — every
extract_dynamic_fields() call passes doc_type="dynamic" to ValidationEngine,
which routes here.

CHANGED — FIXED in this revision: the previous version constructed
ValidationResult with the wrong keyword set (rule_name instead of rule;
missing field, expected, actual, severity — all required, no defaults on
this dataclass). Every construction call was raising TypeError, and
ValidationEngine._load_rules()'s try/except was silently swallowing it —
rules_run showed 3 but results were always empty, with no visible error.
Confirmed against the real validation/rules/base.py contract:

    ValidationResult(field, rule, rule_code, status, expected, actual,
                      message, severity, blocking)
    BaseRule.validate(fields: dict) -> list[ValidationResult], never raises.
"""

from validation.rules.base import BaseRule, ValidationResult


class ListItemConsistencyRule(BaseRule):
    """
    For any field whose value is a list of objects, flag items that are
    missing sub-fields present on other items in the same list — a common
    failure mode where the LLM extracts an early item fully but degrades
    on later ones in a long repeating structure (e.g. a resume with 6+
    past roles, where item 5 loses the office location).

    One ValidationResult per item with at least one missing sub-field.
    """

    def get_name(self) -> str:
        return "List Item Consistency"

    def get_code(self) -> str:
        return "GENERIC_001"

    def validate(self, fields: dict) -> list[ValidationResult]:
        results = []
        for field_name, value in fields.items():
            if not isinstance(value, list) or not value or not isinstance(value[0], dict):
                continue

            # Union of keys across all items, excluding the injected
            # _verification block (derived metadata, not an extracted field).
            all_keys = set()
            for item in value:
                if isinstance(item, dict):
                    all_keys.update(k for k in item.keys() if k != "_verification")

            for i, item in enumerate(value):
                if not isinstance(item, dict):
                    continue
                missing = [
                    k for k in all_keys
                    if k != "_verification" and item.get(k) in (None, "", [])
                ]
                if missing:
                    results.append(ValidationResult(
                        field=f"{field_name}[{i}]",
                        rule=self.get_name(),
                        rule_code=self.get_code(),
                        status="WARNING",
                        expected="all sub-fields populated",
                        actual=f"missing: {', '.join(missing)}",
                        message=f"{field_name}[{i}] is missing: {', '.join(missing)}",
                        severity="WARNING",
                        blocking=False,
                    ))
        return results


class TopLevelCompletenessRule(BaseRule):
    """
    Flags top-level fields that came back entirely null/empty — a generic
    completeness signal that works for any schema, independent of doc type
    or field names. One ValidationResult per empty field.
    """

    def get_name(self) -> str:
        return "Field Completeness"

    def get_code(self) -> str:
        return "GENERIC_002"

    def validate(self, fields: dict) -> list[ValidationResult]:
        results = []
        for field_name, value in fields.items():
            if value in (None, "", []):
                results.append(ValidationResult(
                    field=field_name,
                    rule=self.get_name(),
                    rule_code=self.get_code(),
                    status="WARNING",
                    expected="non-empty value",
                    actual="empty or null",
                    message=f"No value found for '{field_name}'",
                    severity="WARNING",
                    blocking=False,
                ))
        return results


class DateVerificationMismatchRule(BaseRule):
    """
    Surfaces items where schemas.validator.verify_dynamic_extraction() found
    a stated duration that disagrees with the computed one (_verification.match
    is explicitly False, not just None/absent). This is the validation-layer
    hook for the verification data — a mismatch here means either the LLM
    misread a date, or the source document's stated tenure is itself wrong.
    """

    def get_name(self) -> str:
        return "Date Verification Mismatch"

    def get_code(self) -> str:
        return "GENERIC_003"

    def validate(self, fields: dict) -> list[ValidationResult]:
        results = []
        for field_name, value in fields.items():
            if not isinstance(value, list):
                continue
            for i, item in enumerate(value):
                if not isinstance(item, dict):
                    continue
                verification = item.get("_verification")
                if verification and verification.get("match") is False:
                    stated = verification.get("total_time_stated")
                    computed = verification.get("total_time_computed")
                    results.append(ValidationResult(
                        field=f"{field_name}[{i}]",
                        rule=self.get_name(),
                        rule_code=self.get_code(),
                        status="WARNING",
                        expected=f"stated duration matches computed ({computed})",
                        actual=f"stated={stated!r}",
                        message=(
                            f"{field_name}[{i}]: stated duration {stated!r} does not match "
                            f"computed {computed!r} (from {verification.get('start_date')}"
                            f"–{verification.get('end_date')})"
                        ),
                        severity="WARNING",
                        blocking=False,
                    ))
        return results


def get_rules() -> list:
    return [
        ListItemConsistencyRule(),
        TopLevelCompletenessRule(),
        DateVerificationMismatchRule(),
    ]