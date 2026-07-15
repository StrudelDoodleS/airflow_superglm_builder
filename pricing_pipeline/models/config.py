from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ValidationSplitConfig:
    method: str = "kfold"
    n_splits: int | None = 5
    test_size: float | None = None
    random_state: int | None = 42
    shuffle: bool = True
    stratify_column: str | None = None
    materialize: bool = False
    column: str | None = None
    train_values: tuple[Any, ...] = ()
    test_values: tuple[Any, ...] = ()

    @classmethod
    def kfold(
        cls,
        *,
        n_splits: int = 5,
        random_state: int | None = 42,
        shuffle: bool = True,
        materialize: bool = False,
    ) -> "ValidationSplitConfig":
        return cls(
            method="kfold",
            n_splits=n_splits,
            random_state=random_state if shuffle else None,
            shuffle=shuffle,
            materialize=materialize,
        )

    @classmethod
    def train_test_split(
        cls,
        *,
        test_size: float = 0.2,
        random_state: int | None = 42,
        shuffle: bool = True,
        stratify_column: str | None = None,
        materialize: bool = False,
    ) -> "ValidationSplitConfig":
        return cls(
            method="train_test_split",
            n_splits=None,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify_column=stratify_column,
            materialize=materialize,
        )

    @classmethod
    def column_kfold(
        cls,
        *,
        column: str,
        materialize: bool = False,
    ) -> "ValidationSplitConfig":
        return cls(
            method="column_kfold",
            n_splits=None,
            test_size=None,
            random_state=None,
            shuffle=False,
            stratify_column=None,
            materialize=materialize,
            column=column,
        )

    @classmethod
    def column_holdout(
        cls,
        *,
        column: str,
        train_values: tuple[Any, ...],
        test_values: tuple[Any, ...],
        materialize: bool = False,
    ) -> "ValidationSplitConfig":
        return cls(
            method="column_holdout",
            n_splits=None,
            test_size=None,
            random_state=None,
            shuffle=False,
            stratify_column=None,
            materialize=materialize,
            column=column,
            train_values=train_values,
            test_values=test_values,
        )


@dataclass(frozen=True)
class ModelBuildConfig:
    model_name: str
    model_label: str
    target_name: str
    model_type: str
    deployment_slot: str
    validation_split: ValidationSplitConfig = field(default_factory=ValidationSplitConfig.kfold)
