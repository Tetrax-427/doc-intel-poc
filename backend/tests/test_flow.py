"""
Standalone verification of the dynamic/nested schema pipeline, without a live
LLM or DB — exercises exactly the code paths that retrieval.extract_dynamic_fields()
and routers/extraction.py wire together:

  1. A user submits a nested SchemaSpec directly (POST /extract nested_schema case)
     -> SchemaSpec.model_validate() must accept it.
  2. spec_to_model() must build a working recursive Pydantic model from it.
  3. A simulated LLM response (what call_llm(response_model=DynamicModel) would
     return) is validated against that model — proves nested list-of-objects
     round-trips correctly through Instructor's response_model contract.
  4. verify_dynamic_extraction() attaches _verification to each past_companies
     item, computed from start_date/end_date — proves computation happens as
     VERIFICATION, not extraction (the LLM output already has no computed field
     in this test's first company, and a deliberately WRONG stated duration in
     the second, to prove mismatch detection).
  5. The response-shaping logic from retrieval.extract_dynamic_fields() is
     replicated here (flat string -> wrapped w/ bbox; list/object -> raw) to
     show the final API response shape.
"""

import json
import sys

sys.path.insert(0, "/home/claude/docintel_updates")

from schemas.dynamic import SchemaSpec, spec_to_model, spec_to_field_descriptions
from schemas.validator import verify_dynamic_extraction, validate_extraction


# ---------------------------------------------------------------------------
# Step 1 — user-submitted (or /extract/nl-generated) nested schema
# ---------------------------------------------------------------------------

nested_schema_payload = {
    "schema_name": "candidate_profile",
    "fields": [
        {"name": "candidate_name", "type": "string", "description": "Full name of the candidate"},
        {
            "name": "past_companies",
            "type": "list",
            "description": "Each company the candidate has worked at",
            "properties": [
                {"name": "name", "type": "string", "description": "Company name"},
                {"name": "start_date", "type": "date", "description": "Start date, YYYY-MM"},
                {"name": "end_date", "type": "date", "description": "End date, YYYY-MM, or null if current"},
                {"name": "place", "type": "string", "description": "Office location"},
                # deliberately include a stated-duration field so we can test
                # the verification match/mismatch path
                {"name": "total_time", "type": "string", "description": "Stated tenure, if written in the document"},
            ],
        },
    ],
}

spec = SchemaSpec.model_validate(nested_schema_payload)
print("STEP 1 — SchemaSpec.model_validate() accepted nested payload: OK")
print(f"  schema_name={spec.schema_name!r}, fields={[f.name for f in spec.fields]}")

# ---------------------------------------------------------------------------
# Step 2 — recursive model builder
# ---------------------------------------------------------------------------

DynamicModel = spec_to_model(spec)
print("\nSTEP 2 — spec_to_model() built:", DynamicModel.__name__)
print("  model fields:", list(DynamicModel.model_fields.keys()))

# ---------------------------------------------------------------------------
# Step 3 — simulate what Instructor/call_llm(response_model=DynamicModel)
#          would return, validated against the built model
# ---------------------------------------------------------------------------

simulated_llm_output = {
    "candidate_name": "Rohan Mehta",
    "past_companies": [
        {
            "name": "TCS",
            "start_date": "2019-06",
            "end_date": "2021-03",
            "place": "Pune",
            "total_time": None,  # not stated in the doc — nothing to compare against
        },
        {
            "name": "Infosys",
            "start_date": "2021-04",
            "end_date": "2023-12",
            "place": "Bangalore",
            "total_time": "3 years",  # deliberately WRONG (actual is 2y 8m) — tests mismatch
        },
    ],
}

validated = DynamicModel.model_validate(simulated_llm_output)
print("\nSTEP 3 — Instructor-style response_model validation: OK")
raw_extracted = validated.model_dump()
print("  raw_extracted:", json.dumps(raw_extracted, indent=2, default=str))

# ---------------------------------------------------------------------------
# Step 4 — deterministic verification (NOT extraction)
# ---------------------------------------------------------------------------

verified = verify_dynamic_extraction(raw_extracted, spec)
print("\nSTEP 4 — verify_dynamic_extraction() attached _verification:")
for item in verified["past_companies"]:
    print(f"  {item['name']}: {item['_verification']}")

assert verified["past_companies"][0]["_verification"]["total_time_computed"] == "1y 9m"
assert verified["past_companies"][0]["_verification"]["match"] is None  # nothing stated
assert verified["past_companies"][1]["_verification"]["total_time_computed"] == "2y 8m"
assert verified["past_companies"][1]["_verification"]["match"] is False  # "3 years" != "2y 8m"
print("  assertions passed: TCS computed=1y 9m (no stated value, match=None); "
      "Infosys computed=2y 8m vs stated '3 years' -> match=False")

# ---------------------------------------------------------------------------
# Step 5 — response shaping (mirrors retrieval.extract_dynamic_fields())
# ---------------------------------------------------------------------------

def fake_find_field_bbox(value, document_id):
    # stand-in for retrieval.find_field_bbox() — no real document/chunks here
    return {"page": 1, "x": 100, "y": 200} if value == "Rohan Mehta" else None

extracted_response = {}
plain_values = {}
for field_name, value in verified.items():
    if isinstance(value, str):
        extracted_response[field_name] = {"value": value, "bbox": fake_find_field_bbox(value, "doc_123")}
        plain_values[field_name] = value
    else:
        extracted_response[field_name] = value
        plain_values[field_name] = value

top_level_fields = {f.name: f.description for f in spec.fields}
validation = validate_extraction(plain_values, top_level_fields)

final_response = {
    "extracted": extracted_response,
    "validation": validation,
    "business_validation": {"doc_type": "dynamic", "note": "see validation/rulesets/generic.py"},
    "schema_used": spec.model_dump(),
}

print("\nSTEP 5 — final /extract response shape:")
print(json.dumps(final_response, indent=2, default=str))

print("\nALL STEPS PASSED.")