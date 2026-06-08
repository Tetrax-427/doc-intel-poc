# backend/validation/rulesets/bank_statement.py

from validation.rules.arithmetic import PositiveAmountRule
from validation.rules.completeness import RequiredFieldsRule, MinLengthRule


def get_rules() -> list:
    return [
        # Completeness
        RequiredFieldsRule(
            required_fields=["account_holder", "bank_name", "statement_period"],
            blocking=True,
        ),
        RequiredFieldsRule(
            required_fields=["opening_balance", "closing_balance"],
            blocking=False,
        ),

        # Amounts should be parseable numbers (can be zero — e.g. new account)
        # Use MinLength to catch extraction artifacts like "N/A" or "***"
        MinLengthRule(field="account_holder", min_length=3, blocking=False),
        MinLengthRule(field="statement_period", min_length=5, blocking=False),
    ]