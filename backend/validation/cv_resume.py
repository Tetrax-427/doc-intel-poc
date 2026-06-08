# backend/validation/rulesets/cv_resume.py

from validation.rules.logic import RegexRule
from validation.rules.completeness import RequiredFieldsRule, NoEmptyListRule, MinLengthRule


def get_rules() -> list:
    return [
        # Completeness — core identity fields are required
        RequiredFieldsRule(
            required_fields=["full_name", "email"],
            blocking=True,
        ),
        RequiredFieldsRule(
            required_fields=["current_title", "total_experience_years"],
            blocking=False,  # warning only — some CVs omit these
        ),

        # Format checks
        RegexRule(
            field="email",
            pattern=r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$",
            description="valid email address",
            blocking=False,
        ),
        RegexRule(
            field="phone",
            pattern=r"^[\+\d\s\-\(\)]{7,20}$",
            description="valid phone number",
            blocking=False,
        ),

        # Content quality
        NoEmptyListRule(field="skills", blocking=False),
        MinLengthRule(field="full_name", min_length=3, blocking=False),
        MinLengthRule(field="education", min_length=10, blocking=False),
    ]