from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ValidationResult:
    """
    Single rule execution result.
    
    status values  : PASS | FAIL | WARNING | SKIPPED
    severity values: ERROR | WARNING | INFO
    blocking       : if True and status==FAIL, workflow cannot proceed
                     without human override.
    """
    field:     str
    rule:      str
    rule_code: str
    status:    str
    expected:  str
    actual:    str
    message:   str
    severity:  str
    blocking:  bool

    def to_dict(self) -> dict:
        return {
            "field":     self.field,
            "rule":      self.rule,
            "rule_code": self.rule_code,
            "status":    self.status,
            "expected":  self.expected,
            "actual":    self.actual,
            "message":   self.message,
            "severity":  self.severity,
            "blocking":  self.blocking,
        }


class BaseRule(ABC):
    """
    Abstract base for all validation rules.

    Rules must be:
    - Stateless: same inputs -> same outputs, no side effects
    - Safe:      never raise -- return [] if the rule cannot be applied
    - Focused:   one rule checks one thing
    """

    @abstractmethod
    def get_code(self) -> str:
        ...

    @abstractmethod
    def get_name(self) -> str:
        ...

    @abstractmethod
    def validate(self, fields: dict) -> list[ValidationResult]:
        """
        Run this rule against the extracted fields dict.
        Return empty list if rule does not apply (required fields absent).
        Never raises.
        """
        ...