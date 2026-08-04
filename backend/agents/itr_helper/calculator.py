"""
backend/agents/itr_helper/calculator.py
------------------------------------------
Pure Python tax calculation over a stitched taxpayer_profile (see
helpers.stitch_taxpayer_profile). No LLM calls here — every number is
computed deterministically from tax_rules.py + the extracted profile,
per the earlier design decision that all calculations must be grounded/
computed, never LLM-invented.

SCOPE (v1): three income heads only — salary (Form16), other-source
interest (FD), and capital gains on listed equity (stocks). Extensibility
design: income heads are aggregated independently by their own
_aggregate_*_income() function into a common IncomeBreakdown shape, and
compute_tax_comparison() sums whatever heads are present. Adding a new
income head later (house property, business income) means:
  1. Add a new _aggregate_<head>_income() function following the same
     shape (returns amounts pre/post regime-specific adjustments).
  2. Add its field to IncomeBreakdown and include it in the totals.
  3. Add any new regime-specific allow/disallow rules to
     tax_rules.RegimeRules (a new field on that dataclass) rather than
     hardcoding a new if/else here.
No restructuring of compute_tax_comparison()'s call shape or return shape
is expected to be needed — new heads are additive.

ASSUMPTIONS (v1, documented so they're easy to find and revisit):
  - Every stock_transactions entry is listed equity (no asset-type field
    captured yet — see schemas.itr's stocks schema). STCG/LTCG classification
    and rates below assume equity throughout.
  - Section 87A rebate applies to slab-taxed income only (salary + other
    sources), not to capital gains tax — matches common practice, though
    this has been a genuinely contested interpretation across notifications;
    revisit if CBDT guidance changes.
  - 80TTA/80TTB (savings-account interest deduction) is NOT modeled yet —
    FD interest is taxed in full under other_sources for both regimes.
  - Multiple Form16 employers in one year: standard deduction and
    professional-tax deduction are each applied ONCE in total (not once
    per employer), gross salary components are summed across employers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from agents.itr_helper.tax_rules import TaxYearRules, get_tax_rules


# ---------------------------------------------------------------------------
# Income aggregation — one function per income head
# ---------------------------------------------------------------------------

@dataclass
class SalaryIncome:
    gross_salary_total: float
    hra_exemption_total: float
    lta_exemption_total: float
    other_exempt_allowance_total: float
    professional_tax_total: float
    chapter_via_deductions_total: float  # 80C/80D/80CCD(1B)/etc, summed across all Form16s


def _entry_gross_salary(entry: dict) -> float:
    """Prefer the explicit gross_salary field; fall back to summing its components."""
    gross = entry.get("gross_salary")
    if gross is not None:
        return float(gross)
    parts = [
        entry.get("salary_section_17_1") or 0,
        entry.get("perquisites_section_17_2") or 0,
        entry.get("profits_in_lieu_section_17_3") or 0,
    ]
    return float(sum(parts))


def _aggregate_salary_income(form16_entries: list[dict]) -> SalaryIncome:
    gross_total = 0.0
    hra_total = 0.0
    lta_total = 0.0
    other_exempt_total = 0.0
    prof_tax_total = 0.0
    chapter_via_total = 0.0

    for entry in form16_entries:
        gross_total += _entry_gross_salary(entry)

        salary_breakup = entry.get("salary_breakup") or {}
        exempt = salary_breakup.get("exempt_allowances") or {}
        hra_total += float(exempt.get("hra") or 0)
        lta_total += float(exempt.get("lta") or 0)
        other_exempt_total += float(exempt.get("other") or 0)
        prof_tax_total += float(salary_breakup.get("professional_tax") or 0)

        for ded in entry.get("deductions") or []:
            chapter_via_total += float(ded.get("amount_claimed") or 0)

    return SalaryIncome(
        gross_salary_total=gross_total,
        hra_exemption_total=hra_total,
        lta_exemption_total=lta_total,
        other_exempt_allowance_total=other_exempt_total,
        professional_tax_total=prof_tax_total,
        chapter_via_deductions_total=chapter_via_total,
    )


def _aggregate_other_source_income(fd_interest_entries: list[dict]) -> float:
    """Sum of FD/savings interest across all certificates. No 80TTA/80TTB yet — see module docstring."""
    return sum(float(fd.get("interest_earned") or 0) for fd in fd_interest_entries)


@dataclass
class CapitalGainsIncome:
    stcg_total: float
    ltcg_total: float


def _parse_date(value) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    return None


def _aggregate_capital_gains(stock_transactions: list[dict], rules: TaxYearRules) -> CapitalGainsIncome:
    stcg_total = 0.0
    ltcg_total = 0.0
    cg_rules = rules.capital_gains

    for txn in stock_transactions:
        buy_date = _parse_date(txn.get("buy_date"))
        sell_date = _parse_date(txn.get("sell_date"))
        buy_value = float(txn.get("buy_value") or 0)
        sell_value = float(txn.get("sell_value") or 0)
        gain = sell_value - buy_value

        is_short_term = True  # default to STCG (higher tax) if dates are missing/unparseable
        if buy_date and sell_date:
            holding_days = (sell_date - buy_date).days
            is_short_term = holding_days <= cg_rules.holding_period_days

        if is_short_term:
            stcg_total += gain
        else:
            ltcg_total += gain

    return CapitalGainsIncome(stcg_total=stcg_total, ltcg_total=ltcg_total)


# ---------------------------------------------------------------------------
# Slab tax computation — regime-agnostic given a RegimeRules
# ---------------------------------------------------------------------------

def _compute_slab_tax(taxable_income: float, rules) -> float:
    """rules: tax_rules.RegimeRules. Standard progressive slab computation."""
    if taxable_income <= 0:
        return 0.0
    tax = 0.0
    lower = 0.0
    for slab in rules.slabs:
        if taxable_income <= lower:
            break
        upper = min(slab.upto, taxable_income)
        tax += (upper - lower) * slab.rate
        lower = slab.upto
    return tax


def _apply_rebate_87a(tax_before_rebate: float, taxable_income: float, rules) -> float:
    if taxable_income <= rules.rebate_87a_income_threshold:
        return max(0.0, tax_before_rebate - rules.rebate_87a_max_amount)
    return tax_before_rebate


def _compute_surcharge(base_tax: float, total_income: float, rules: TaxYearRules) -> float:
    rate = 0.0
    for slab in rules.surcharge_slabs:
        if total_income <= slab.upto:
            rate = slab.rate
            break
    else:
        rate = rules.surcharge_slabs[-1].rate if rules.surcharge_slabs else 0.0
    return base_tax * rate


# ---------------------------------------------------------------------------
# Capital gains tax — same for both regimes
# ---------------------------------------------------------------------------

def _compute_capital_gains_tax(cg: CapitalGainsIncome, rules: TaxYearRules) -> float:
    cg_rules = rules.capital_gains
    stcg_tax = max(0.0, cg.stcg_total) * cg_rules.stcg_rate
    taxable_ltcg = max(0.0, cg.ltcg_total - cg_rules.ltcg_exemption_amount)
    ltcg_tax = taxable_ltcg * cg_rules.ltcg_rate
    return stcg_tax + ltcg_tax


# ---------------------------------------------------------------------------
# Per-regime computation
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    regime: str  # "old" | "new"
    taxable_salary_income: float
    taxable_other_source_income: float
    slab_taxable_income: float
    tax_before_rebate: float
    tax_after_rebate: float
    capital_gains_tax: float
    surcharge: float
    cess: float
    total_tax_payable: float


def _compute_regime(
    regime_name: str,
    salary: SalaryIncome,
    other_source_income: float,
    cg: CapitalGainsIncome,
    rules: TaxYearRules,
) -> RegimeResult:
    regime_rules = rules.new_regime if regime_name == "new" else rules.old_regime

    taxable_salary = salary.gross_salary_total
    if regime_rules.allows_hra_lta_exemption:
        taxable_salary -= (
            salary.hra_exemption_total
            + salary.lta_exemption_total
            + salary.other_exempt_allowance_total
        )
    taxable_salary -= regime_rules.standard_deduction
    if regime_rules.allows_professional_tax_deduction:
        taxable_salary -= salary.professional_tax_total
    taxable_salary = max(0.0, taxable_salary)

    slab_taxable_income = taxable_salary + other_source_income
    if regime_rules.allows_chapter_via_deductions:
        slab_taxable_income -= salary.chapter_via_deductions_total
    slab_taxable_income = max(0.0, slab_taxable_income)

    tax_before_rebate = _compute_slab_tax(slab_taxable_income, regime_rules)
    tax_after_rebate = _apply_rebate_87a(tax_before_rebate, slab_taxable_income, regime_rules)

    capital_gains_tax = _compute_capital_gains_tax(cg, rules)

    total_income_for_surcharge = slab_taxable_income + max(0.0, cg.stcg_total) + max(0.0, cg.ltcg_total)
    base_for_surcharge = tax_after_rebate + capital_gains_tax
    surcharge = _compute_surcharge(base_for_surcharge, total_income_for_surcharge, rules)

    cess = (tax_after_rebate + capital_gains_tax + surcharge) * rules.cess_rate
    total_tax_payable = tax_after_rebate + capital_gains_tax + surcharge + cess

    return RegimeResult(
        regime=regime_name,
        taxable_salary_income=round(taxable_salary, 2),
        taxable_other_source_income=round(other_source_income, 2),
        slab_taxable_income=round(slab_taxable_income, 2),
        tax_before_rebate=round(tax_before_rebate, 2),
        tax_after_rebate=round(tax_after_rebate, 2),
        capital_gains_tax=round(capital_gains_tax, 2),
        surcharge=round(surcharge, 2),
        cess=round(cess, 2),
        total_tax_payable=round(total_tax_payable, 2),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_tax_comparison(taxpayer_profile: dict, assessment_year: str | None = None) -> dict:
    """
    Computes tax under both regimes from a stitched taxpayer_profile
    (see helpers.stitch_taxpayer_profile) and returns a side-by-side
    comparison with a recommendation.

    Returns:
        {
            "assessment_year": str,
            "old_regime": {...RegimeResult fields...},
            "new_regime": {...RegimeResult fields...},
            "recommended_regime": "old" | "new",
            "savings_amount": float,  # always >= 0, how much the recommended regime saves vs the other
        }
    """
    rules = get_tax_rules(assessment_year)

    salary = _aggregate_salary_income(taxpayer_profile.get("form16_entries", []))
    other_source_income = _aggregate_other_source_income(taxpayer_profile.get("fd_interest_entries", []))
    cg = _aggregate_capital_gains(taxpayer_profile.get("stock_transactions", []), rules)

    old_result = _compute_regime("old", salary, other_source_income, cg, rules)
    new_result = _compute_regime("new", salary, other_source_income, cg, rules)

    if new_result.total_tax_payable <= old_result.total_tax_payable:
        recommended = "new"
        savings = old_result.total_tax_payable - new_result.total_tax_payable
    else:
        recommended = "old"
        savings = new_result.total_tax_payable - old_result.total_tax_payable

    return {
        "assessment_year": rules.assessment_year,
        "old_regime": old_result.__dict__,
        "new_regime": new_result.__dict__,
        "recommended_regime": recommended,
        "savings_amount": round(savings, 2),
    }