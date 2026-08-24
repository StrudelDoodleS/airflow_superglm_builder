from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from superglm.editor import EditorSession

from pricing_pipeline.workbench.core import Candidate

_POLICY_FORMAT = "pricing-manual-adjustment-policy-v1"


def _required_text(value: Any, field_name: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    return cleaned


def _factor(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("factor must be a positive finite number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError("factor must be a positive finite number")
    if resolved == 1.0:
        raise ValueError("factor must change the selected relativity")
    return resolved


@dataclass(frozen=True)
class ManualAdjustmentRule:
    """One relative business adjustment that can be replayed on a new base model."""

    feature: str
    factor: float
    reason: str
    levels: tuple[str, ...] | None = None
    x_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature", _required_text(self.feature, "feature"))
        object.__setattr__(self, "factor", _factor(self.factor))
        object.__setattr__(self, "reason", _required_text(self.reason, "rule reason"))
        if (self.levels is None) == (self.x_range is None):
            raise ValueError("provide exactly one of levels or x_range")
        if self.levels is not None:
            levels = tuple(_required_text(level, "level") for level in self.levels)
            if not levels:
                raise ValueError("levels must contain at least one value")
            if len(levels) != len(set(levels)):
                raise ValueError("levels must not contain duplicates")
            object.__setattr__(self, "levels", levels)
        if self.x_range is not None:
            if len(self.x_range) != 2:
                raise ValueError("x_range must contain exactly two values")
            start, stop = (float(value) for value in self.x_range)
            if not math.isfinite(start) or not math.isfinite(stop):
                raise ValueError("x_range values must be finite")
            object.__setattr__(self, "x_range", (min(start, stop), max(start, stop)))

    @classmethod
    def multiply_levels(
        cls,
        feature: str,
        levels: Iterable[str],
        factor: float,
        *,
        reason: str,
    ) -> ManualAdjustmentRule:
        return cls(
            feature=feature,
            levels=tuple(levels),
            factor=factor,
            reason=reason,
        )

    @classmethod
    def multiply_range(
        cls,
        feature: str,
        start: float,
        stop: float,
        factor: float,
        *,
        reason: str,
    ) -> ManualAdjustmentRule:
        return cls(
            feature=feature,
            x_range=(start, stop),
            factor=factor,
            reason=reason,
        )

    def apply(self, session: EditorSession) -> None:
        if self.levels is not None:
            session.select_levels(self.feature, list(self.levels))
        else:
            assert self.x_range is not None
            session.select_x(self.feature, *self.x_range)
        session.shift(self.feature, math.log(self.factor))

    def to_payload(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "feature": self.feature,
            "levels": None if self.levels is None else list(self.levels),
            "operation": "MULTIPLY",
            "reason": self.reason,
            "x_range": None if self.x_range is None else list(self.x_range),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ManualAdjustmentRule:
        if str(payload.get("operation") or "").upper() != "MULTIPLY":
            raise ValueError("manual adjustment operation must be MULTIPLY")
        levels = payload.get("levels")
        x_range = payload.get("x_range")
        return cls(
            feature=payload.get("feature", ""),
            factor=payload.get("factor"),
            reason=payload.get("reason", ""),
            levels=None if levels is None else tuple(levels),
            x_range=None if x_range is None else tuple(x_range),
        )


@dataclass(frozen=True)
class ManualAdjustmentPolicy:
    """Versioned relative rules, independent of any one fitted coefficient vector."""

    name: str
    version: int
    reason: str
    rules: tuple[ManualAdjustmentRule, ...]
    carry_forward: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, "policy name"))
        object.__setattr__(self, "reason", _required_text(self.reason, "policy reason"))
        if isinstance(self.version, bool) or int(self.version) < 1:
            raise ValueError("policy version must be a positive integer")
        object.__setattr__(self, "version", int(self.version))
        object.__setattr__(self, "carry_forward", bool(self.carry_forward))
        rules = tuple(self.rules)
        if not rules:
            raise ValueError("manual adjustment policy requires at least one rule")
        if not all(isinstance(rule, ManualAdjustmentRule) for rule in rules):
            raise TypeError("rules must contain ManualAdjustmentRule values")
        _reject_overlapping_rules(rules)
        object.__setattr__(self, "rules", rules)

    @classmethod
    def from_rows(
        cls,
        *,
        name: str,
        version: int,
        reason: str,
        rows: Iterable[Mapping[str, Any]],
        carry_forward: bool = True,
    ) -> ManualAdjustmentPolicy:
        return cls(
            name=name,
            version=version,
            reason=reason,
            carry_forward=carry_forward,
            rules=tuple(
                ManualAdjustmentRule.from_payload({"operation": "MULTIPLY", **dict(row)})
                for row in rows
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "carry_forward": self.carry_forward,
            "format": _POLICY_FORMAT,
            "name": self.name,
            "reason": self.reason,
            "rules": [rule.to_payload() for rule in self.rules],
            "version": self.version,
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.to_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ManualAdjustmentPolicy:
        if payload.get("format") != _POLICY_FORMAT:
            raise ValueError("manual adjustment policy has an unsupported format")
        raw_rules = payload.get("rules")
        if not isinstance(raw_rules, list):
            raise TypeError("manual adjustment policy rules must be a list")
        return cls(
            name=payload.get("name", ""),
            version=payload.get("version"),
            reason=payload.get("reason", ""),
            carry_forward=payload.get("carry_forward", True),
            rules=tuple(ManualAdjustmentRule.from_payload(rule) for rule in raw_rules),
        )

    def table(self) -> pd.DataFrame:
        rows = []
        for rule in self.rules:
            rows.append(
                {
                    "Feature": rule.feature,
                    "Scope": "Levels" if rule.levels is not None else "Range",
                    "Selection": (
                        ", ".join(rule.levels)
                        if rule.levels is not None
                        else f"{rule.x_range[0]:g} to {rule.x_range[1]:g}"
                    ),
                    "Multiplier": rule.factor,
                    "Change": f"{(rule.factor - 1.0):+.2%}",
                    "Reason": rule.reason,
                }
            )
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class ManualEditReview:
    candidate: Candidate
    policy: ManualAdjustmentPolicy
    editor_session: EditorSession
    edited_model: Any
    impact: Mapping[str, Any]

    @property
    def rules(self) -> pd.DataFrame:
        return self.policy.table()


def _reject_overlapping_rules(rules: tuple[ManualAdjustmentRule, ...]) -> None:
    for index, left in enumerate(rules):
        for right in rules[index + 1 :]:
            if left.feature != right.feature:
                continue
            if left.levels is not None and right.levels is not None:
                overlap = sorted(set(left.levels) & set(right.levels))
                if overlap:
                    raise ValueError(
                        f"manual adjustment rules overlap for {left.feature!r}: {overlap}"
                    )
            elif left.x_range is not None and right.x_range is not None:
                if max(left.x_range[0], right.x_range[0]) <= min(left.x_range[1], right.x_range[1]):
                    raise ValueError(f"manual adjustment ranges overlap for {left.feature!r}")
            else:
                raise ValueError(
                    f"manual adjustment rules mix level and range scopes for {left.feature!r}"
                )


def _predict(model: Any, candidate: Candidate) -> np.ndarray:
    kwargs: dict[str, Any] = {}
    if candidate.bundle.offset is not None:
        kwargs["offset"] = candidate.bundle.offset
    values = np.asarray(model.predict(candidate.bundle.X, **kwargs), dtype=np.float64)
    if values.ndim != 1 or len(values) != len(candidate.bundle.X):
        raise ValueError("manual adjustment preview produced invalid predictions")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("manual adjustment preview produced invalid predictions")
    return values


def _impact(candidate: Candidate, edited_model: Any) -> dict[str, Any]:
    baseline = _predict(candidate.bundle.fitted_model, candidate)
    edited = _predict(edited_model, candidate)
    raw_weights = candidate.bundle.export_weight
    if raw_weights is None:
        weights = np.ones(len(baseline), dtype=np.float64)
        weight_source = "unit"
    else:
        weights = np.asarray(raw_weights, dtype=np.float64)
        weight_source = candidate.bundle.export_weight_name or "export_weight"
    if weights.ndim != 1 or len(weights) != len(baseline):
        raise ValueError("manual adjustment preview weights do not align with predictions")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("manual adjustment preview weights must be finite and non-negative")
    baseline_mean = float(np.average(baseline, weights=weights))
    edited_mean = float(np.average(edited, weights=weights))
    ratio = edited_mean / baseline_mean if baseline_mean else math.nan
    row_ratio = np.divide(
        edited,
        baseline,
        out=np.ones_like(edited),
        where=baseline != 0,
    )
    return {
        "Source package": candidate.package_version,
        "Source kind": candidate.technical.get("model_kind"),
        "Data as-at": candidate.technical.get("data_as_of_date"),
        "Rows reviewed": len(baseline),
        "Weight source": weight_source,
        "Baseline weighted mean": baseline_mean,
        "Edited weighted mean": edited_mean,
        "Portfolio multiplier": ratio,
        "Portfolio change": ratio - 1.0,
        "Changed rows": int(np.count_nonzero(~np.isclose(row_ratio, 1.0))),
        "Minimum row multiplier": float(row_ratio.min()),
        "Maximum row multiplier": float(row_ratio.max()),
    }


def apply_manual_adjustment_policy(
    candidate: Candidate,
    policy: ManualAdjustmentPolicy,
) -> ManualEditReview:
    """Apply relative policy rules to one exact candidate and produce review evidence."""
    if not isinstance(candidate, Candidate):
        raise TypeError("candidate must come from open_candidate() or open_deployed_candidate()")
    if not isinstance(policy, ManualAdjustmentPolicy):
        raise TypeError("policy must be a ManualAdjustmentPolicy")
    session = EditorSession.from_model(
        candidate.bundle.fitted_model,
        train_data=(
            candidate.bundle.X,
            candidate.bundle.y,
            candidate.bundle.sample_weight,
            candidate.bundle.offset,
        ),
        cv_report=candidate.bundle.cv_report,
    )
    for rule in policy.rules:
        rule.apply(session)
    edited_model = session.to_model()
    return ManualEditReview(
        candidate=candidate,
        policy=policy,
        editor_session=session,
        edited_model=edited_model,
        impact=_impact(candidate, edited_model),
    )


def manual_adjustment_policy_from_candidate(
    candidate: Candidate,
) -> ManualAdjustmentPolicy:
    """Recover the replayable policy embedded in a published MANUAL_EDIT package."""
    if not isinstance(candidate, Candidate):
        raise TypeError("candidate must come from open_candidate() or open_deployed_candidate()")
    if str(candidate.technical.get("model_kind") or "").upper() != "MANUAL_EDIT":
        raise ValueError("candidate is not a MANUAL_EDIT package")
    raw_metadata = candidate.technical.get("revision_metadata_json")
    if isinstance(raw_metadata, str):
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError as exc:
            raise ValueError("manual edit revision metadata is not valid JSON") from exc
    elif isinstance(raw_metadata, Mapping):
        metadata = dict(raw_metadata)
    else:
        raise TypeError("manual edit package has no revision metadata")
    edit_metadata = metadata.get("edit_metadata")
    if not isinstance(edit_metadata, Mapping):
        raise TypeError("manual edit package has no structured edit metadata")
    payload = edit_metadata.get("manual_adjustment_policy")
    if not isinstance(payload, Mapping):
        raise TypeError("manual edit package has no replayable adjustment policy")
    policy = ManualAdjustmentPolicy.from_payload(payload)
    expected_sha256 = edit_metadata.get("manual_adjustment_policy_sha256")
    if expected_sha256 != policy.sha256:
        raise ValueError("manual adjustment policy SHA-256 verification failed")
    return policy


__all__ = [
    "ManualAdjustmentPolicy",
    "ManualAdjustmentRule",
    "ManualEditReview",
    "apply_manual_adjustment_policy",
    "manual_adjustment_policy_from_candidate",
]
