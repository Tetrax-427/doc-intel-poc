# backend/validation/rulesets/invoice.py

from validation.rules.arithmetic import SumRule, TotalConsistencyRule, PositiveAmountRule
from validation.rules.logic import DateOrderRule, NotFutureDateRule
from validation.rules.completeness import RequiredFieldsRule, NoEmptyListRule


def get_rules() -> list:
    return [
        # Completeness
        RequiredFieldsRule(
            required_fields=["vendor_name", "invoice_number", "invoice_date", "total_amount"],
            blocking=True,
        ),
        NoEmptyListRule(field="line_items", blocking=False),

        # Arithmetic
        SumRule(item_field="line_items", total_field="subtotal", blocking=True),
        TotalConsistencyRule(),
        PositiveAmountRule(field="total_amount", blocking=True),

        # Date logic
        NotFutureDateRule(field="invoice_date", blocking=False),
        DateOrderRule(start_field="invoice_date", end_field="due_date", blocking=False),
    ]