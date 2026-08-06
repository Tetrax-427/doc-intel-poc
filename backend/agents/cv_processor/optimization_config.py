"""
backend/agents/cv_processor/optimization_config.py
------------------------------------------------------
Tunable knobs for CV Processor's batching/chunking behavior. Centralized
here so call-volume/prompt-size tradeoffs can be adjusted without touching
pipeline logic in agent.py or tools/scoring.py.
"""

# ---- verify_claims ----
# Candidates per LLM call (still evaluated independently within the batch).
VERIFY_CLAIMS_BATCH_SIZE = 5

# Verification only runs on a subset of candidates, chosen from the initial
# ranking (merge_scores runs BEFORE verify_claims in the pipeline).
# If the requester asked for a specific top-N, verify top N * this multiplier.
VERIFY_CLAIMS_TOP_N_MULTIPLIER = 1.5
# If no specific N was requested (full ranking), verify this fraction of
# all candidates instead (0.35 = top 35%).
VERIFY_CLAIMS_TOP_PERCENT = 0.35

# ---- score_skills ----
# Target candidate x skill-group pairs per LLM call. Drives the chunk size:
# candidates_per_call = max(1, SCORE_SKILLS_PAIRS_PER_CALL // n_skill_groups).
# Total calls = ceil(n_skill_groups * n_candidates / SCORE_SKILLS_PAIRS_PER_CALL).
SCORE_SKILLS_PAIRS_PER_CALL = 20

# ---- compute_adjusted_scores ----
# Candidates per LLM call, default (each call covers ALL of a candidate's
# flagged claims across every criterion, not one call per claim).
ADJUSTED_SCORES_BATCH_SIZE = 5
# Hard cap on total calls for this stage. If the number of flagged
# candidates would need more calls than this at the default batch size,
# batch size grows instead (never exceed this many calls).
ADJUSTED_SCORES_MAX_CALLS = 10