"""
backend/agents/itr_helper/tax_rules.py
------------------------------------------
Versioned Indian income-tax rules, keyed by assessment year. Swappable by
design: TAX_RULES is a plain dict of AY -> TaxYearRules; adding next year's
rules (or correcting this year's after a mid-year notification) means
adding/editing one dict entry here — nothing else in the codebase changes,
since calculator.py only ever calls get_tax_rules(ay).

SCOPE (v1): salary income (Form16), other-source interest income (FD),
and capital gains on listed equity (stocks) only — no house property,
business income, or loss carry-forward rules yet. See calculator.py's
module docstring for how new income heads get added later without
restructuring this file or calculator.py's public functions.

Rules below are for AY 2026-27 (FY 2025-26), per Union Budget 2025's
announced new-regime slabs and Budget 2024's capital gains rate changes
carried forward. Verify against the current CBDT notification before
relying on this for real filings — tax rules can be revised after
publication.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SlabRule:
    """One income tax slab: taxed at `rate` for the portion of income up to `upto`."""
    upto: float  # use float('inf') for the top/unbounded slab
    rate: float  # e.g. 0.05 for 5%


@dataclass(frozen=True)
class RegimeRules:
    """Everything needed to compute slab-taxed income tax for one regime (old or new)."""
    slabs: list[SlabRule]
    standard_deduction: float
    rebate_87a_income_threshold: float  # taxable income at/below this -> full rebate
    rebate_87a_max_amount: float        # rebate capped at this amount
    # Old regime allows HRA/LTA exemptions, professional tax deduction, and
    # Chapter VI-A deductions (80C/80D/80CCD(1B)/etc); new regime doesn't
    # (with narrow exceptions like 80CCD(2), not modeled yet — v1 scope).
    allows_hra_lta_exemption: bool
    allows_professional_tax_deduction: bool
    allows_chapter_via_deductions: bool


@dataclass(frozen=True)
class CapitalGainsRules:
    """
    v1 scope: listed equity only (all stock_transactions are assumed to be
    listed equity — no asset-type field captured yet, see calculator.py).
    Same for both regimes — regime choice doesn't affect capital gains
    tax treatment, only slab-income treatment.
    """
    stcg_rate: float                 # short-term (holding period <= holding_period_days)
    ltcg_rate: float                 # long-term, above the exemption
    ltcg_exemption_amount: float     # LTCG below this, per year, is tax-free
    holding_period_days: int         # <= this => STCG, > this => LTCG


@dataclass(frozen=True)
class SurchargeSlab:
    upto: float
    rate: float


@dataclass(frozen=True)
class TaxYearRules:
    assessment_year: str
    old_regime: RegimeRules
    new_regime: RegimeRules
    capital_gains: CapitalGainsRules
    cess_rate: float
    # Surcharge slabs on total income, applied identically to both regimes
    # in v1 (real old-regime surcharge tops out at 37% above 5Cr vs new
    # regime's 25% cap — this is a documented v1 simplification, capped at
    # 25% for both; revisit if a run's total income actually exceeds 5Cr).
    surcharge_slabs: list[SurchargeSlab] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AY 2026-27 (FY 2025-26)
# ---------------------------------------------------------------------------

_AY_2026_27 = TaxYearRules(
    assessment_year="2026-27",
    new_regime=RegimeRules(
        slabs=[
            SlabRule(400_000, 0.00),
            SlabRule(800_000, 0.05),
            SlabRule(1_200_000, 0.10),
            SlabRule(1_600_000, 0.15),
            SlabRule(2_000_000, 0.20),
            SlabRule(2_400_000, 0.25),
            SlabRule(float("inf"), 0.30),
        ],
        standard_deduction=75_000,
        rebate_87a_income_threshold=1_200_000,
        rebate_87a_max_amount=60_000,
        allows_hra_lta_exemption=False,
        allows_professional_tax_deduction=False,
        allows_chapter_via_deductions=False,
    ),
    old_regime=RegimeRules(
        slabs=[
            SlabRule(250_000, 0.00),
            SlabRule(500_000, 0.05),
            SlabRule(1_000_000, 0.20),
            SlabRule(float("inf"), 0.30),
        ],
        standard_deduction=50_000,
        rebate_87a_income_threshold=500_000,
        rebate_87a_max_amount=12_500,
        allows_hra_lta_exemption=True,
        allows_professional_tax_deduction=True,
        allows_chapter_via_deductions=True,
    ),
    capital_gains=CapitalGainsRules(
        stcg_rate=0.20,
        ltcg_rate=0.125,
        ltcg_exemption_amount=125_000,
        holding_period_days=365,
    ),
    cess_rate=0.04,
    surcharge_slabs=[
        SurchargeSlab(5_000_000, 0.00),
        SurchargeSlab(10_000_000, 0.10),
        SurchargeSlab(20_000_000, 0.15),
        SurchargeSlab(float("inf"), 0.25),
    ],
)


# Registry — add a new AY's TaxYearRules here to support it. Nothing else
# in calculator.py or the agent stages needs to change.
TAX_RULES: dict[str, TaxYearRules] = {
    "2026-27": _AY_2026_27,
}

DEFAULT_ASSESSMENT_YEAR = "2026-27"


class UnknownAssessmentYearError(Exception):
    pass


def get_tax_rules(assessment_year: str | None = None) -> TaxYearRules:
    """Returns the TaxYearRules for the given AY, or DEFAULT_ASSESSMENT_YEAR if None."""
    ay = assessment_year or DEFAULT_ASSESSMENT_YEAR
    rules = TAX_RULES.get(ay)
    if rules is None:
        raise UnknownAssessmentYearError(
            f"No tax rules registered for AY '{ay}'. Available: {list(TAX_RULES.keys())}"
        )
    return rules