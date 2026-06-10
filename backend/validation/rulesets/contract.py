from validation.rules.logic import DateOrderRule, NotFutureDateRule
from validation.rules.completeness import RequiredFieldsRule, NoEmptyListRule, MinLengthRule


def get_rules() -> list:
    return [
        # Completeness — both parties and effective date are the minimum
        RequiredFieldsRule(
            required_fields=["party_1", "party_2", "effective_date", "agreement_type"],
            blocking=True,
        ),

        # Date logic
        NotFutureDateRule(field="effective_date", blocking=False),
        DateOrderRule(
            start_field="effective_date",
            end_field="expiry_date",
            blocking=True,
        ),

        # Content quality
        NoEmptyListRule(field="key_obligations", blocking=False),
        MinLengthRule(field="party_1", min_length=2, blocking=False),
        MinLengthRule(field="party_2", min_length=2, blocking=False),
        MinLengthRule(field="governing_law", min_length=3, blocking=False),
    ]