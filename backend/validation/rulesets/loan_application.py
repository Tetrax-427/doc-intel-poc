from validation.rules.arithmetic import PositiveAmountRule
from validation.rules.logic import ConditionalRequiredRule, NotFutureDateRule
from validation.rules.completeness import RequiredFieldsRule, MinLengthRule


def get_rules() -> list:
    return [
        # Completeness — applicant identity and loan basics are required
        RequiredFieldsRule(
            required_fields=[
                "applicant_name", "loan_type",
                "loan_amount_requested", "loan_tenure",
            ],
            blocking=True,
        ),
        RequiredFieldsRule(
            required_fields=["annual_income", "employment_type"],
            blocking=False,
        ),

        # Arithmetic
        PositiveAmountRule(field="loan_amount_requested", blocking=True),
        PositiveAmountRule(field="annual_income", blocking=False),

        # Conditional: salaried applicants must have employer name
        ConditionalRequiredRule(
            condition_field="employment_type",
            condition_value="salaried",
            required_field="employer_name",
            blocking=False,
        ),

        # Date checks
        NotFutureDateRule(field="applicant_dob", blocking=False),

        # Content quality
        MinLengthRule(field="applicant_name", min_length=3, blocking=False),
    ]