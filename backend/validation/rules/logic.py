import re
from datetime import datetime
from validation.rules.base import BaseRule, ValidationResult

DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
    "%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%d %b %Y",
    "%Y/%m/%d", "%Y",
]


def _parse_date(val) -> datetime | None:
    """Try multiple date formats. Returns None if all fail."""
    if not val:
        return None
    s = str(val).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


class DateOrderRule(BaseRule):
    """
    Verify that start_field is chronologically before end_field.

    Example: invoice_date < due_date
             effective_date < expiry_date
    """

    def __init__(self, start_field: str, end_field: str, blocking: bool = True):
        self.start_field = start_field
        self.end_field = end_field
        self.blocking = blocking

    def get_code(self) -> str:
        return "VAL_201"

    def get_name(self) -> str:
        return f"date_order({self.start_field}<{self.end_field})"

    def validate(self, fields: dict) -> list[ValidationResult]:
        start_val = fields.get(self.start_field)
        end_val   = fields.get(self.end_field)

        if not start_val or not end_val:
            return []  # one or both dates missing — skip

        start = _parse_date(start_val)
        end   = _parse_date(end_val)

        if not start or not end:
            return [ValidationResult(
                field=self.end_field,
                rule=self.get_name(),
                rule_code=self.get_code(),
                status="WARNING",
                expected="parseable dates",
                actual=f"{start_val}, {end_val}",
                message=f"Could not parse dates: '{start_val}', '{end_val}'",
                severity="WARNING",
                blocking=False,
            )]

        if start < end:
            return [ValidationResult(
                field=self.end_field,
                rule=self.get_name(),
                rule_code=self.get_code(),
                status="PASS",
                expected=f"{self.start_field} < {self.end_field}",
                actual=f"{start_val} < {end_val}",
                message=f"Date order valid: {self.start_field} before {self.end_field} \u2713",
                severity="INFO",
                blocking=False,
            )]
        return [ValidationResult(
            field=self.end_field,
            rule=self.get_name(),
            rule_code=self.get_code(),
            status="FAIL",
            expected=f"{self.start_field} before {self.end_field}",
            actual=f"{start_val} >= {end_val}",
            message=(
                f"{self.start_field} ({start_val}) must be before "
                f"{self.end_field} ({end_val})"
            ),
            severity="ERROR",
            blocking=self.blocking,
        )]


class ConditionalRequiredRule(BaseRule):
    """
    If condition_field == condition_value, then required_field must be present.

    Example: if employment_type == "salaried" then employer_name is required
    """

    def __init__(
        self,
        condition_field: str,
        condition_value,
        required_field: str,
        blocking: bool = True,
    ):
        self.condition_field = condition_field
        self.condition_value = condition_value
        self.required_field  = required_field
        self.blocking        = blocking

    def get_code(self) -> str:
        return "VAL_202"

    def get_name(self) -> str:
        return (
            f"if_{self.condition_field}=="
            f"{self.condition_value}_then_{self.required_field}_required"
        )

    def validate(self, fields: dict) -> list[ValidationResult]:
        actual_value = fields.get(self.condition_field)

        # Normalise comparison — both to lowercase string if string
        def norm(v):
            return str(v).lower().strip() if v is not None else None

        if norm(actual_value) != norm(self.condition_value):
            return []  # condition not met — rule does not apply

        present = bool(fields.get(self.required_field))

        if present:
            return [ValidationResult(
                field=self.required_field,
                rule=self.get_name(),
                rule_code=self.get_code(),
                status="PASS",
                expected="present",
                actual="present",
                message=f"{self.required_field} present as required \u2713",
                severity="INFO",
                blocking=False,
            )]
        return [ValidationResult(
            field=self.required_field,
            rule=self.get_name(),
            rule_code=self.get_code(),
            status="FAIL",
            expected="present",
            actual="missing",
            message=(
                f"{self.required_field} is required when "
                f"{self.condition_field} = {self.condition_value}"
            ),
            severity="ERROR",
            blocking=self.blocking,
        )]


class RegexRule(BaseRule):
    """
    Verify that a field value matches a regex pattern.

    Example: GSTIN must match 15-char alphanumeric pattern
             PAN must match AAAAA9999A pattern
    """

    def __init__(
        self,
        field: str,
        pattern: str,
        description: str,
        blocking: bool = True,
    ):
        self.field       = field
        self.pattern     = pattern
        self.description = description
        self.blocking    = blocking
        self._compiled   = re.compile(pattern, re.IGNORECASE)

    def get_code(self) -> str:
        return "VAL_203"

    def get_name(self) -> str:
        return f"regex({self.field})"

    def validate(self, fields: dict) -> list[ValidationResult]:
        value = fields.get(self.field)
        if not value:
            return []  # field absent — completeness rule handles this

        s = str(value).strip()
        if self._compiled.match(s):
            return [ValidationResult(
                field=self.field,
                rule=self.get_name(),
                rule_code=self.get_code(),
                status="PASS",
                expected=self.description,
                actual=s,
                message=f"{self.field} format valid \u2713",
                severity="INFO",
                blocking=False,
            )]
        return [ValidationResult(
            field=self.field,
            rule=self.get_name(),
            rule_code=self.get_code(),
            status="FAIL",
            expected=self.description,
            actual=s,
            message=f"{self.field} format invalid: expected {self.description}, got '{s}'",
            severity="ERROR",
            blocking=self.blocking,
        )]


class NotFutureDateRule(BaseRule):
    """
    Verify that a date field is not in the future.

    Useful for: invoice_date, dob, statement_period start dates.
    """

    def __init__(self, field: str, blocking: bool = False):
        self.field    = field
        self.blocking = blocking

    def get_code(self) -> str:
        return "VAL_204"

    def get_name(self) -> str:
        return f"not_future({self.field})"

    def validate(self, fields: dict) -> list[ValidationResult]:
        raw = fields.get(self.field)
        if not raw:
            return []

        parsed = _parse_date(raw)
        if not parsed:
            return []  # regex rule handles format errors

        now = datetime.utcnow()
        if parsed <= now:
            return [ValidationResult(
                field=self.field,
                rule=self.get_name(),
                rule_code=self.get_code(),
                status="PASS",
                expected="not in future",
                actual=str(raw),
                message=f"{self.field} is not a future date \u2713",
                severity="INFO",
                blocking=False,
            )]
        return [ValidationResult(
            field=self.field,
            rule=self.get_name(),
            rule_code=self.get_code(),
            status="FAIL",
            expected="date <= today",
            actual=str(raw),
            message=f"{self.field} ({raw}) is a future date",
            severity="WARNING",
            blocking=self.blocking,
        )]