from validation.rules.arithmetic import GSTConsistencyRule, PositiveAmountRule
from validation.rules.logic import RegexRule
from validation.rules.completeness import RequiredFieldsRule


def get_rules() -> list:
    return [
        # Completeness
        RequiredFieldsRule(
            required_fields=["gstin", "legal_name", "filing_period", "return_type"],
            blocking=True,
        ),
        RequiredFieldsRule(
            required_fields=["total_tax_liability"],
            blocking=False,
        ),

        # Format — GSTIN is 15 chars: 2 digit state code + 10 char PAN + 3 chars
        RegexRule(
            field="gstin",
            pattern=r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$",
            description="valid 15-character GSTIN",
            blocking=True,
        ),

        # Arithmetic
        GSTConsistencyRule(),
        PositiveAmountRule(field="taxable_turnover", blocking=False),
        PositiveAmountRule(field="total_tax_liability", blocking=False),
    ]