"""
validation/rulesets/generic.py — Schema-agnostic rules for dynamically
generated (user-defined) extraction schemas, where field names aren't known
in advance so the doc-type-specific rulesets (invoice.py, cv_resume.py, etc.)
can't apply.

Wired in via validation/engine.py's RULESET_MAP under "dynamic" — every
extract_dynamic_fields() call passes doc_type="dynamic" to ValidationEngine,
which routes here.

NOTE: written against the BaseRule/ValidationResult contract inferred from
validation/engine.py (rule.get_name(), rule.get_code(), rule.validate(dict)
-> list[ValidationResult], and ValidationResult.status/.blocking/.to_dict()).
Please verify these two imports against the actual validation/rules/base.py
before running — that file wasn't in the files reviewed for this change, so
the constructor signature below is a best-effort match to the existing
ruleset pattern rather than a confirmed one.
"""

from validation.rules.base import BaseRule, ValidationResult


class ListItemConsistencyRule(BaseRule):
    """
    For any field whose value is a list of objects, flag items that are
    missing sub-fields present on other items in the same list — a common
    failure mode where the LLM extracts an early item fully but degrades
    on later ones in a long repeating structure (e.g. a resume with 6+
    past roles, where item 5 loses the office location).
    """

    def get_name(self) -> str:
        return "List Item Consistency"

    def get_code(self) -> str:
        return "GENERIC_001"

    def validate(self, extracted: dict) -> list[ValidationResult]:
        results = []
        for field_name, value in extracted.items():
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
                        rule_code=self.get_code(),
                        rule_name=self.get_name(),
                        status="WARNING",
                        blocking=False,
                        message=f"{field_name}[{i}] missing: {', '.join(missing)}",
                    ))
        return results


class TopLevelCompletenessRule(BaseRule):
    """
    Flags top-level fields that came back entirely null/empty — a generic
    completeness signal that works for any schema, independent of doc type
    or field names.
    """

    def get_name(self) -> str:
        return "Field Completeness"

    def get_code(self) -> str:
        return "GENERIC_002"

    def validate(self, extracted: dict) -> list[ValidationResult]:
        results = []
        empty_fields = [k for k, v in extracted.items() if v in (None, "", [])]
        if empty_fields:
            results.append(ValidationResult(
                rule_code=self.get_code(),
                rule_name=self.get_name(),
                status="WARNING",
                blocking=False,
                message=f"No value found for: {', '.join(empty_fields)}",
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

    def validate(self, extracted: dict) -> list[ValidationResult]:
        results = []
        for field_name, value in extracted.items():
            if not isinstance(value, list):
                continue
            for i, item in enumerate(value):
                if not isinstance(item, dict):
                    continue
                verification = item.get("_verification")
                if verification and verification.get("match") is False:
                    results.append(ValidationResult(
                        rule_code=self.get_code(),
                        rule_name=self.get_name(),
                        status="WARNING",
                        blocking=False,
                        message=(
                            f"{field_name}[{i}]: stated duration "
                            f"'{verification.get('total_time_stated')}' does not match "
                            f"computed '{verification.get('total_time_computed')}' "
                            f"from {verification.get('start_date')}–{verification.get('end_date')}"
                        ),
                    ))
        return results


def get_rules() -> list:
    return [
        ListItemConsistencyRule(),
        TopLevelCompletenessRule(),
        DateVerificationMismatchRule(),
    ]