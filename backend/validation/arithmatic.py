# backend/validation/rules/arithmetic.py

from validation.rules.base import BaseRule, ValidationResult


def _parse_amount(v) -> float | None:
    """
    Parse a monetary value from various string formats.
    Handles: "1,23,456.78", "Rs. 1000", "1000 INR", "$500", "500.00"
    Returns None if unparseable.
    """
    if v is None:
        return None
    try:
        cleaned = (
            str(v)
            .replace(",", "")
            .replace("₹", "").replace("Rs.", "").replace("Rs", "")
            .replace("$", "").replace("€", "").replace("£", "")
            .replace("INR", "").replace("USD", "").replace("EUR", "")
            .strip()
        )
        return float(cleaned)
    except (ValueError, TypeError):
        return None


class SumRule(BaseRule):
    """
    Verify that the sum of amounts in a list field equals a total field.

    Example: sum(line_items[].amount) == subtotal

    Config:
        item_field:   name of the list field in extracted dict
        total_field:  name of the total field to compare against
        amount_keys:  list of dict keys to try for each item's amount
                      (tries each in order, uses first found)
        blocking:     whether a FAIL blocks the workflow
    """

    def __init__(
        self,
        item_field: str,
        total_field: str,
        amount_keys: list[str] = None,
        blocking: bool = True,
    ):
        self.item_field = item_field
        self.total_field = total_field
        self.amount_keys = amount_keys or ["amount", "price", "value", "total", "rate"]
        self.blocking = blocking

    def get_code(self) -> str:
        return "VAL_101"

    def get_name(self) -> str:
        return f"sum({self.item_field})=={self.total_field}"

    def validate(self, fields: dict) -> list[ValidationResult]:
        items = fields.get(self.item_field)
        total_raw = fields.get(self.total_field)

        if not items or total_raw is None:
            return []  # rule does not apply

        amounts = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    for k in self.amount_keys:
                        if k in item:
                            v = _parse_amount(item[k])
                            if v is not None:
                                amounts.append(v)
                            break
                else:
                    v = _parse_amount(item)
                    if v is not None:
                        amounts.append(v)

        if not amounts:
            return []  # nothing parseable — skip rather than false-fail

        computed = round(sum(amounts), 2)
        expected = _parse_amount(total_raw)
        if expected is None:
            return []

        if abs(computed - expected) <= 0.01:
            return [ValidationResult(
                field=self.total_field,
                rule=self.get_name(),
                rule_code=self.get_code(),
                status="PASS",
                expected=str(expected),
                actual=str(computed),
                message=f"Sum of {self.item_field} matches {self.total_field} \u2713",
                severity="INFO",
                blocking=False,
            )]
        return [ValidationResult(
            field=self.total_field,
            rule=self.get_name(),
            rule_code=self.get_code(),
            status="FAIL",
            expected=str(expected),
            actual=str(computed),
            message=(
                f"Sum of {self.item_field} ({computed}) "
                f"\u2260 {self.total_field} ({expected})"
            ),
            severity="ERROR",
            blocking=self.blocking,
        )]


class TotalConsistencyRule(BaseRule):
    """
    Verify: subtotal + tax_amount == total_amount

    Tolerates small floating-point differences (<= 0.01).
    Always blocking — a mismatched total is a data integrity error.
    """

    def get_code(self) -> str:
        return "VAL_102"

    def get_name(self) -> str:
        return "subtotal+tax==total"

    def validate(self, fields: dict) -> list[ValidationResult]:
        subtotal = _parse_amount(fields.get("subtotal"))
        tax      = _parse_amount(fields.get("tax_amount"))
        total    = _parse_amount(fields.get("total_amount"))

        if subtotal is None or tax is None or total is None:
            return []  # not enough fields — skip

        computed = round(subtotal + tax, 2)

        if abs(computed - total) <= 0.01:
            return [ValidationResult(
                field="total_amount",
                rule=self.get_name(),
                rule_code=self.get_code(),
                status="PASS",
                expected=str(total),
                actual=str(computed),
                message="subtotal + tax_amount == total_amount \u2713",
                severity="INFO",
                blocking=False,
            )]
        return [ValidationResult(
            field="total_amount",
            rule=self.get_name(),
            rule_code=self.get_code(),
            status="FAIL",
            expected=str(total),
            actual=str(computed),
            message=(
                f"subtotal ({subtotal}) + tax_amount ({tax}) "
                f"= {computed} \u2260 total_amount ({total})"
            ),
            severity="ERROR",
            blocking=True,
        )]


class GSTConsistencyRule(BaseRule):
    """
    Verify: cgst + sgst + igst == total_tax_liability

    Indian GST documents often have split tax components.
    Non-blocking warning — LLM extraction of tax splits is imprecise.
    """

    def get_code(self) -> str:
        return "VAL_103"

    def get_name(self) -> str:
        return "cgst+sgst+igst==total_tax"

    def validate(self, fields: dict) -> list[ValidationResult]:
        cgst  = _parse_amount(fields.get("cgst"))
        sgst  = _parse_amount(fields.get("sgst"))
        igst  = _parse_amount(fields.get("igst"))
        total = _parse_amount(fields.get("total_tax_liability"))

        # Need at least one of cgst/igst and the total
        if total is None:
            return []
        if cgst is None and igst is None:
            return []

        computed = round((cgst or 0) + (sgst or 0) + (igst or 0), 2)

        if abs(computed - total) <= 0.01:
            return [ValidationResult(
                field="total_tax_liability",
                rule=self.get_name(),
                rule_code=self.get_code(),
                status="PASS",
                expected=str(total),
                actual=str(computed),
                message="GST component totals match \u2713",
                severity="INFO",
                blocking=False,
            )]
        return [ValidationResult(
            field="total_tax_liability",
            rule=self.get_name(),
            rule_code=self.get_code(),
            status="FAIL",
            expected=str(total),
            actual=str(computed),
            message=(
                f"CGST ({cgst or 0}) + SGST ({sgst or 0}) + IGST ({igst or 0}) "
                f"= {computed} \u2260 total_tax_liability ({total})"
            ),
            severity="WARNING",
            blocking=False,  # warning only — extraction splits are often imprecise
        )]


class PositiveAmountRule(BaseRule):
    """
    Verify that a monetary field is a positive number.

    Useful for: total_amount, loan_amount_requested, annual_income etc.
    """

    def __init__(self, field: str, blocking: bool = True):
        self.field = field
        self.blocking = blocking

    def get_code(self) -> str:
        return "VAL_104"

    def get_name(self) -> str:
        return f"{self.field}>0"

    def validate(self, fields: dict) -> list[ValidationResult]:
        raw = fields.get(self.field)
        if raw is None:
            return []

        value = _parse_amount(raw)
        if value is None:
            return [ValidationResult(
                field=self.field,
                rule=self.get_name(),
                rule_code=self.get_code(),
                status="WARNING",
                expected="numeric > 0",
                actual=str(raw),
                message=f"{self.field} could not be parsed as a number",
                severity="WARNING",
                blocking=False,
            )]

        if value > 0:
            return [ValidationResult(
                field=self.field,
                rule=self.get_name(),
                rule_code=self.get_code(),
                status="PASS",
                expected="> 0",
                actual=str(value),
                message=f"{self.field} is positive \u2713",
                severity="INFO",
                blocking=False,
            )]
        return [ValidationResult(
            field=self.field,
            rule=self.get_name(),
            rule_code=self.get_code(),
            status="FAIL",
            expected="> 0",
            actual=str(value),
            message=f"{self.field} must be positive, got {value}",
            severity="ERROR",
            blocking=self.blocking,
        )]