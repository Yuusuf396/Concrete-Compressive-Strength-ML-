"""Feature engineering helpers shared by classical and NN experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import FEATURE_COLUMNS, SplitBundle


DERIVED_FEATURES = {
    "water_cement_ratio": lambda df: df["water"] / df["cement"].clip(lower=1e-3),
    "binder_total": lambda df: df[["cement", "slag", "fly_ash"]].sum(axis=1),
    "agg_ratio": lambda df: df["coarse_aggregate"] / df["fine_aggregate"].clip(lower=1e-3),
}


@dataclass
class FeatureTransform:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    transformer: Pipeline


def make_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df[FEATURE_COLUMNS].copy()
    for col, fn in DERIVED_FEATURES.items():
        enriched[col] = fn(df)
    return enriched


def inject_features(bundle: SplitBundle, feature_frame: pd.DataFrame) -> SplitBundle:
    return SplitBundle(
        X_train=feature_frame.loc[bundle.X_train.index],
        X_val=feature_frame.loc[bundle.X_val.index],
        X_test=feature_frame.loc[bundle.X_test.index],
        y_train=bundle.y_train,
        y_val=bundle.y_val,
        y_test=bundle.y_test,
    )


def scale_features(splits: SplitBundle) -> FeatureTransform:
    feature_names = list(splits.X_train.columns)
    transformer = Pipeline(steps=[("scaler", StandardScaler())])
    X_train_scaled = transformer.fit_transform(splits.X_train)
    X_val_scaled = transformer.transform(splits.X_val)
    X_test_scaled = transformer.transform(splits.X_test)
    return FeatureTransform(
        X_train=pd.DataFrame(X_train_scaled, columns=feature_names, index=splits.X_train.index),
        X_val=pd.DataFrame(X_val_scaled, columns=feature_names, index=splits.X_val.index),
        X_test=pd.DataFrame(X_test_scaled, columns=feature_names, index=splits.X_test.index),
        transformer=transformer,
    )


def prepare_feature_splits(df: pd.DataFrame, splits: SplitBundle) -> FeatureTransform:
    feature_frame = make_feature_frame(df)
    enriched_split = SplitBundle(
        X_train=feature_frame.loc[splits.X_train.index],
        X_val=feature_frame.loc[splits.X_val.index],
        X_test=feature_frame.loc[splits.X_test.index],
        y_train=splits.y_train,
        y_val=splits.y_val,
        y_test=splits.y_test,
    )
    return scale_features(enriched_split)
