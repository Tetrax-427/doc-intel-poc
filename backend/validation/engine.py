import importlib
from core.logger import get_logger

logger = get_logger("validation")

# Maps doc_type strings (from classify_document) to ruleset module paths.
# Unknown types return an empty rule list — no false failures on unrecognised docs.
#
# CHANGED in this phase: added "dynamic" -> validation.rulesets.generic.
# retrieval.extract_dynamic_fields() passes doc_type="dynamic" for every
# nested/user-defined schema extraction (since field names aren't known in
# advance, the doc-type-specific rulesets below don't apply).
RULESET_MAP: dict[str, str] = {
    "invoice":          "validation.rulesets.invoice",
    "receipt":          "validation.rulesets.invoice",   # reuse invoice ruleset
    "cv_resume":        "validation.rulesets.cv_resume",
    "resume":           "validation.rulesets.cv_resume",
    "cv":               "validation.rulesets.cv_resume",
    "gst_return":       "validation.rulesets.gst_return",
    "gstr-1":           "validation.rulesets.gst_return",
    "gstr-3b":          "validation.rulesets.gst_return",
    "contract":         "validation.rulesets.contract",
    "agreement":        "validation.rulesets.contract",
    "nda":              "validation.rulesets.contract",
    "bank_statement":   "validation.rulesets.bank_statement",
    "loan_application": "validation.rulesets.loan_application",
    "dynamic":          "validation.rulesets.generic",
}


class ValidationEngine:
    """
    Runs business logic validation rules against LLM-extracted fields.

    Usage:
        engine = ValidationEngine()
        result = engine.validate(extracted_fields, doc_type="invoice")

    The returned dict matches Contract 4 in CONTRACTS.md exactly.
    """

    def validate(self, extracted_fields: dict, doc_type: str = "general") -> dict:
        """
        Load the ruleset for doc_type and run all rules against extracted_fields.

        Args:
            extracted_fields: Dict of field_name -> extracted value (from LLM).
            doc_type:         Classified document type string.

        Returns:
            Contract 4 shaped dict with is_valid, counts, and full results list.
            Always returns a valid dict — never raises.
        """
        rules = self._load_rules(doc_type)

        if not rules:
            logger.info(
                "No ruleset for doc_type — skipping validation",
                doc_type=doc_type,
            )
            return self._empty_result(doc_type)

        all_results = []
        for rule in rules:
            try:
                results = rule.validate(extracted_fields)
                all_results.extend(results)
            except Exception as e:
                logger.error(
                    "Rule execution failed — skipping rule",
                    rule=rule.get_name(),
                    rule_code=rule.get_code(),
                    error=str(e),
                )

        passed   = [r for r in all_results if r.status == "PASS"]
        failed   = [r for r in all_results if r.status == "FAIL"]
        warnings = [r for r in all_results if r.status == "WARNING"]
        blocking = [r for r in failed if r.blocking]

        logger.info(
            "Validation complete",
            doc_type=doc_type,
            rules_run=len(rules),
            passed=len(passed),
            failed=len(failed),
            warnings=len(warnings),
            blocking=len(blocking),
        )

        return {
            "doc_type":          doc_type,
            "rules_run":         len(rules),
            "passed":            len(passed),
            "failed":            len(failed),
            "warnings":          len(warnings),
            "blocking_failures": len(blocking),
            "is_valid":          len(blocking) == 0,
            "results": [r.to_dict() for r in all_results],
        }

    def _load_rules(self, doc_type: str) -> list:
        """
        Dynamically import the ruleset module for doc_type.
        Returns empty list if no ruleset exists or import fails.
        """
        # Normalise — LLM output may have spaces or mixed case
        normalised = doc_type.lower().strip()
        module_path = RULESET_MAP.get(normalised)

        if not module_path:
            return []

        try:
            module = importlib.import_module(module_path)
            return module.get_rules()
        except Exception as e:
            logger.error(
                "Failed to load ruleset",
                doc_type=doc_type,
                module=module_path,
                error=str(e),
            )
            return []

    def _empty_result(self, doc_type: str) -> dict:
        return {
            "doc_type":          doc_type,
            "rules_run":         0,
            "passed":            0,
            "failed":            0,
            "warnings":          0,
            "blocking_failures": 0,
            "is_valid":          True,
            "results":           [],
        }