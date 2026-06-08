# backend/validation/rules/completeness.py

from validation.rules.base import BaseRule, ValidationResult


class RequiredFieldsRule(BaseRule):
    """
    Verify that all required fields are present and non-empty.

    Returns one FAIL result per missing field — so callers can see
    exactly which fields are absent rather than just "something missing".

    Config:
        required_fields: list of field names that must be present
        blocking:        whether missing fields block the workflow
    """

    def __init__(self, required_fields: list[str], blocking: bool = True):
        self.required_fields = required_fields
        self.blocking        = blocking

    def get_code(self) -> str:
        return "VAL_301"

    def get_name(self) -> str:
        return "required_fields_present"

    def validate(self, fields: dict) -> list[ValidationResult]:
        results = []
        for field in self.required_fields:
            value = fields.get(field)
            is_present = bool(value) and str(value).strip() not in ("", "null", "none", "n/a")

            if is_present:
                results.append(ValidationResult(
                    field=field,
                    rule=self.get_name(),
                    rule_code=self.get_code(),
                    status="PASS",
                    expected="present",
                    actual="present",
                    message=f"{field} is present \u2713",
                    severity="INFO",
                    blocking=False,
                ))
            else:
                results.append(ValidationResult(
                    field=field,
                    rule=self.get_name(),
                    rule_code=self.get_code(),
                    status="FAIL",
                    expected="present",
                    actual="missing",
                    message=f"Required field '{field}' is missing or empty",
                    severity="ERROR",
                    blocking=self.blocking,
                ))
        return results


class NoEmptyListRule(BaseRule):
    """
    Verify that a list field contains at least one item.

    Useful for: line_items on invoices, skills on CVs, key_obligations on contracts.
    """

    def __init__(self, field: str, blocking: bool = False):
        self.field    = field
        self.blocking = blocking

    def get_code(self) -> str:
        return "VAL_302"

    def get_name(self) -> str:
        return f"non_empty_list({self.field})"

    def validate(self, fields: dict) -> list[ValidationResult]:
        value = fields.get(self.field)
        if value is None:
            return []  # field absent — RequiredFieldsRule handles this

        is_non_empty = isinstance(value, list) and len(value) > 0

        if is_non_empty:
            return [ValidationResult(
                field=self.field,
                rule=self.get_name(),
                rule_code=self.get_code(),
                status="PASS",
                expected=">= 1 item",
                actual=f"{len(value)} items",
                message=f"{self.field} has {len(value)} item(s) \u2713",
                severity="INFO",
                blocking=False,
            )]
        return [ValidationResult(
            field=self.field,
            rule=self.get_name(),
            rule_code=self.get_code(),
            status="FAIL",
            expected=">= 1 item",
            actual="0 items",
            message=f"{self.field} must contain at least one item",
            severity="WARNING",
            blocking=self.blocking,
        )]


class MinLengthRule(BaseRule):
    """
    Verify that a string field meets a minimum character length.

    Useful for catching placeholder extractions like "N/A", "unknown",
    single-word summaries, or truncated values.
    """

    def __init__(self, field: str, min_length: int, blocking: bool = False):
        self.field      = field
        self.min_length = min_length
        self.blocking   = blocking

    def get_code(self) -> str:
        return "VAL_303"

    def get_name(self) -> str:
        return f"min_length({self.field},{self.min_length})"

    def validate(self, fields: dict) -> list[ValidationResult]:
        value = fields.get(self.field)
        if not value:
            return []  # absent — RequiredFieldsRule handles this

        s = str(value).strip()
        if len(s) >= self.min_length:
            return [ValidationResult(
                field=self.field,
                rule=self.get_name(),
                rule_code=self.get_code(),
                status="PASS",
                expected=f">= {self.min_length} chars",
                actual=f"{len(s)} chars",
                message=f"{self.field} meets minimum length \u2713",
                severity="INFO",
                blocking=False,
            )]
        return [ValidationResult(
            field=self.field,
            rule=self.get_name(),
            rule_code=self.get_code(),
            status="FAIL",
            expected=f">= {self.min_length} chars",
            actual=f"{len(s)} chars",
            message=(
                f"{self.field} is too short "
                f"({len(s)} chars, minimum {self.min_length})"
            ),
            severity="WARNING",
            blocking=self.blocking,
        )]