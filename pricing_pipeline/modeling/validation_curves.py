from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import numpy as np
from superglm.links import (
    CauchitLink,
    CloglogLink,
    IdentityLink,
    InverseLink,
    InverseSquaredLink,
    Link,
    LogLink,
    LogitLink,
    NegativeBinomialLink,
    PowerLink,
    ProbitLink,
    SqrtLink,
)

from pricing_pipeline.models.spec import ValidationCurvePoint


_SUPPORTED_FAMILIES = frozenset({"continuous", "level"})
_UNSUPPORTED_FAMILIES = frozenset({"interaction", "offset", "surface"})
_STATELESS_LINK_TYPES = frozenset(
    {
        CauchitLink,
        CloglogLink,
        IdentityLink,
        InverseLink,
        InverseSquaredLink,
        LogLink,
        LogitLink,
        ProbitLink,
        SqrtLink,
    }
)


class ValidationCurvePayloadError(ValueError):
    """Raised internally when a supported upstream curve payload is malformed."""


@dataclass(frozen=True)
class ValidationCurveCapture:
    """Atomic normalized validation-curve capture outcome."""

    status: Literal["COMPLETE", "UNAVAILABLE"]
    reason: str | None
    points: tuple[ValidationCurvePoint, ...]

    def __post_init__(self) -> None:
        if self.status not in {"COMPLETE", "UNAVAILABLE"}:
            raise ValueError("status must be COMPLETE or UNAVAILABLE")
        if not isinstance(self.points, tuple) or any(
            not isinstance(point, ValidationCurvePoint) for point in self.points
        ):
            raise ValueError("points must be a tuple of ValidationCurvePoint records")
        reason = self.reason
        if reason is not None:
            if not isinstance(reason, str):
                raise ValueError("reason must be a string or null")
            reason = " ".join(reason.split())
            if not reason:
                raise ValueError("reason must be non-empty when supplied")
            if len(reason) > 500:
                raise ValueError("reason must contain at most 500 characters")
            object.__setattr__(self, "reason", reason)
        if self.status == "COMPLETE":
            if reason is not None or not self.points:
                raise ValueError("COMPLETE requires a null reason and at least one point")
        elif reason is None or self.points:
            raise ValueError("UNAVAILABLE requires a reason and zero points")


@dataclass(frozen=True)
class _NormalizedTerm:
    name: str
    family: Literal["continuous", "level"]
    domain: tuple[float | str, ...]
    support: tuple[float, ...]
    reference_index: int
    link_curves: tuple[tuple[float, ...], ...]


def normalize_validation_curves(
    curve_similarity: Any,
    *,
    estimators: Sequence[Any] | None,
    fold_count: int,
) -> ValidationCurveCapture:
    """Normalize supported SuperGLM main-effect link curves into scalar points."""
    try:
        return _normalize_validation_curves(
            curve_similarity,
            estimators=estimators,
            fold_count=fold_count,
        )
    except ValidationCurvePayloadError as exc:
        return _unavailable_capture("validation curve capture unavailable", exc)
    except Exception as exc:  # Defensive boundary around untrusted upstream diagnostics.
        wrapped = ValidationCurvePayloadError(
            f"unexpected malformed payload ({type(exc).__name__}: {exc})"
        )
        return _unavailable_capture("validation curve capture unavailable", wrapped)


def validation_curve_capture_failure(exc: Exception) -> ValidationCurveCapture:
    """Describe an estimator-enabled CV failure without retaining the exception."""
    return _unavailable_capture("validation curve capture failed", exc)


def _normalize_validation_curves(
    curve_similarity: Any,
    *,
    estimators: Sequence[Any] | None,
    fold_count: int,
) -> ValidationCurveCapture:
    if isinstance(fold_count, bool) or not isinstance(fold_count, Integral) or fold_count <= 0:
        raise ValidationCurvePayloadError("fold_count must be a positive integer")
    links = _resolved_links(estimators, fold_count=int(fold_count))
    log_link = type(links[0]) is LogLink

    if not isinstance(curve_similarity, Mapping) or not curve_similarity:
        raise ValidationCurvePayloadError("curve_similarity must be a non-empty mapping")

    supported_payloads: list[tuple[str, Literal["continuous", "level"], Mapping[str, Any]]] = []
    for raw_term_name, raw_payload in curve_similarity.items():
        if not isinstance(raw_term_name, str) or not raw_term_name.strip():
            raise ValidationCurvePayloadError("curve term names must be non-empty strings")
        term_name = raw_term_name
        if not isinstance(raw_payload, Mapping):
            raise ValidationCurvePayloadError(f"term {term_name!r} payload must be a mapping")
        family = raw_payload.get("family")
        if not isinstance(family, str):
            raise ValidationCurvePayloadError(f"term {term_name!r} must declare a string family")
        if family in _UNSUPPORTED_FAMILIES:
            continue
        if family not in _SUPPORTED_FAMILIES:
            raise ValidationCurvePayloadError(
                f"term {term_name!r} declares unknown family {family!r}"
            )
        supported_payloads.append((term_name, family, raw_payload))

    if not supported_payloads:
        raise ValidationCurvePayloadError("no supported main-effect entries were returned")

    expected_labels = tuple(f"fold_{index}" for index in range(int(fold_count)))
    terms = tuple(
        _normalize_term(term_name, family, payload, expected_labels=expected_labels)
        for term_name, family, payload in sorted(
            supported_payloads,
            key=lambda entry: entry[0],
        )
    )

    points: list[ValidationCurvePoint] = []
    for fold_index in range(int(fold_count)):
        for term in terms:
            curve = term.link_curves[fold_index]
            reference_eta = curve[term.reference_index]
            for point_index, (domain_value, support_value, eta) in enumerate(
                zip(term.domain, term.support, curve, strict=True),
                start=1,
            ):
                rebased_eta = eta - reference_eta
                if not math.isfinite(rebased_eta):
                    raise ValidationCurvePayloadError(
                        f"term {term.name!r} fold_{fold_index} rebased link curve is non-finite"
                    )
                try:
                    relativity = math.exp(rebased_eta) if log_link else None
                except OverflowError as exc:
                    raise ValidationCurvePayloadError(
                        f"term {term.name!r} fold_{fold_index} relativity is non-finite"
                    ) from exc
                if relativity is not None and not math.isfinite(relativity):
                    raise ValidationCurvePayloadError(
                        f"term {term.name!r} fold_{fold_index} relativity is non-finite"
                    )
                numeric = term.family == "continuous"
                points.append(
                    ValidationCurvePoint(
                        validation_split_no=fold_index + 1,
                        term_name=term.name,
                        point_no=point_index,
                        point_kind="NUMERIC" if numeric else "LEVEL",
                        x_numeric=float(domain_value) if numeric else None,
                        level_text=None if numeric else domain_value,
                        eta_contribution=rebased_eta,
                        relativity=relativity,
                        support_value=support_value,
                        reference_value=(
                            float(term.domain[term.reference_index]) if numeric else None
                        ),
                        reference_level=None if numeric else term.domain[term.reference_index],
                    )
                )
    return ValidationCurveCapture(status="COMPLETE", reason=None, points=tuple(points))


def _resolved_links(
    estimators: Sequence[Any] | None,
    *,
    fold_count: int,
) -> tuple[Any, ...]:
    if estimators is None or isinstance(estimators, str | bytes):
        raise ValidationCurvePayloadError(
            "fold estimators must be a sequence matching the validation split count"
        )
    try:
        estimator_values = tuple(estimators)
    except TypeError as exc:
        raise ValidationCurvePayloadError("fold estimators must be a sequence") from exc
    if len(estimator_values) != fold_count:
        raise ValidationCurvePayloadError(
            "fold estimator count must exactly match the validation split count"
        )
    links = tuple(getattr(estimator, "_link", None) for estimator in estimator_values)
    if any(link is None for link in links):
        raise ValidationCurvePayloadError("every fold estimator must expose a resolved _link")
    signatures = tuple(_link_semantic_signature(link) for link in links)
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValidationCurvePayloadError(
            "all fold estimator link classes and parameters must agree"
        )
    return links


def _link_semantic_signature(link: Any) -> tuple[Any, ...]:
    if not isinstance(link, Link):
        raise ValidationCurvePayloadError(
            "every resolved _link must satisfy the SuperGLM Link protocol"
        )
    link_type = type(link)
    if link_type in _STATELESS_LINK_TYPES:
        return (link_type,)
    if link_type is PowerLink:
        parameter_name = "power"
    elif link_type is NegativeBinomialLink:
        parameter_name = "theta"
    else:
        raise ValidationCurvePayloadError(
            f"unsupported resolved link class {link_type.__module__}.{link_type.__qualname__}"
        )
    try:
        parameter_value = float(getattr(link, parameter_name))
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValidationCurvePayloadError(
            f"resolved {link_type.__name__} link parameter {parameter_name} must be finite"
        ) from exc
    if not math.isfinite(parameter_value):
        raise ValidationCurvePayloadError(
            f"resolved {link_type.__name__} link parameter {parameter_name} must be finite"
        )
    return (link_type, parameter_name, parameter_value)


def _normalize_term(
    term_name: str,
    family: Literal["continuous", "level"],
    payload: Mapping[str, Any],
    *,
    expected_labels: tuple[str, ...],
) -> _NormalizedTerm:
    domain_payload = _mapping(payload.get("domain"), f"term {term_name!r} domain")
    support_payload = _mapping(payload.get("support"), f"term {term_name!r} support")
    domain_key = "x" if family == "continuous" else "levels"
    if domain_key not in domain_payload:
        raise ValidationCurvePayloadError(f"term {term_name!r} domain is missing {domain_key!r}")
    if domain_key not in support_payload:
        raise ValidationCurvePayloadError(f"term {term_name!r} support is missing {domain_key!r}")

    if family == "continuous":
        domain: tuple[float | str, ...] = _finite_numeric_vector(
            domain_payload[domain_key],
            f"term {term_name!r} domain x",
        )
        support_domain = _finite_numeric_vector(
            support_payload[domain_key],
            f"term {term_name!r} support x",
        )
    else:
        domain = _unique_level_vector(
            domain_payload[domain_key],
            f"term {term_name!r} domain levels",
        )
        support_domain = _unique_level_vector(
            support_payload[domain_key],
            f"term {term_name!r} support levels",
        )
    if domain != support_domain:
        raise ValidationCurvePayloadError(
            f"term {term_name!r} support domain must exactly match its common domain"
        )

    if "density" not in support_payload:
        raise ValidationCurvePayloadError(f"term {term_name!r} support is missing 'density'")
    support = _finite_numeric_vector(
        support_payload["density"],
        f"term {term_name!r} support density",
    )
    if len(support) != len(domain):
        raise ValidationCurvePayloadError(
            f"term {term_name!r} support density length must match its domain"
        )
    if any(value < 0.0 for value in support):
        raise ValidationCurvePayloadError(f"term {term_name!r} support density must be nonnegative")
    curves_payload = _mapping(payload.get("curves"), f"term {term_name!r} curves")
    if "link" not in curves_payload:
        raise ValidationCurvePayloadError(f"term {term_name!r} curves are missing 'link'")
    link_payload = _mapping(
        curves_payload["link"],
        f"term {term_name!r} link curves",
    )
    if set(link_payload) != set(expected_labels):
        actual_labels = sorted(str(label) for label in link_payload)
        raise ValidationCurvePayloadError(
            f"term {term_name!r} link curve labels must be exactly "
            f"{list(expected_labels)!r}; got {actual_labels!r}"
        )
    link_curves: list[tuple[float, ...]] = []
    for label in expected_labels:
        curve = _finite_numeric_vector(
            link_payload[label],
            f"term {term_name!r} {label} link curve",
        )
        if len(curve) != len(domain):
            raise ValidationCurvePayloadError(
                f"term {term_name!r} {label} link curve length must match its domain"
            )
        link_curves.append(curve)

    return _NormalizedTerm(
        name=term_name,
        family=family,
        domain=domain,
        support=support,
        reference_index=int(np.argmax(np.asarray(support, dtype=np.float64))),
        link_curves=tuple(link_curves),
    )


def _mapping(value: Any, label: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise ValidationCurvePayloadError(f"{label} must be a mapping")
    return value


def _finite_numeric_vector(value: Any, label: str) -> tuple[float, ...]:
    try:
        raw = np.asarray(value)
    except Exception as exc:
        raise ValidationCurvePayloadError(f"{label} must be a one-dimensional vector") from exc
    if raw.ndim != 1 or raw.size == 0 or np.issubdtype(raw.dtype, np.bool_):
        raise ValidationCurvePayloadError(
            f"{label} must be a non-empty one-dimensional numeric vector"
        )
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationCurvePayloadError(f"{label} must contain only numbers") from exc
    if not np.isfinite(numeric).all():
        raise ValidationCurvePayloadError(f"{label} must contain only finite values")
    return tuple(float(item) for item in numeric)


def _unique_level_vector(value: Any, label: str) -> tuple[str, ...]:
    try:
        raw = np.asarray(value, dtype=object)
    except Exception as exc:
        raise ValidationCurvePayloadError(f"{label} must be a one-dimensional vector") from exc
    if raw.ndim != 1 or raw.size == 0:
        raise ValidationCurvePayloadError(
            f"{label} must be a non-empty one-dimensional level vector"
        )
    levels: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ValidationCurvePayloadError(f"{label} must contain only string levels")
        if not item.strip():
            raise ValidationCurvePayloadError(f"{label} must contain only non-empty string levels")
        levels.append(item)
    if len(set(levels)) != len(levels):
        raise ValidationCurvePayloadError(f"{label} must contain unique levels")
    return tuple(levels)


def _unavailable_capture(
    prefix: str,
    exc: Exception,
) -> ValidationCurveCapture:
    detail = " ".join(str(exc).split()) or "no detail"
    reason = f"{prefix}: {type(exc).__name__}: {detail}"[:500]
    return ValidationCurveCapture(status="UNAVAILABLE", reason=reason, points=())
