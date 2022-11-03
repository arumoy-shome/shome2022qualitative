"""Module for data manipulation."""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

METRICS = [
    "statistical_parity_difference",
    "disparate_impact",
    "average_abs_odds_difference",
    "true_positive_rate_difference",
]


def pivot_frame(data: pd.DataFrame, values: str) -> pd.DataFrame:
    """Pivot given data."""
    return data.pivot(
        index="iteration",
        columns="model",
        values=values,
    )


def process(frame):
    """Wrapper for all preprocessing methods.

    Args:
        frame: pandas.DataFrame
    """

    def round(frame, cols):
        """Round columns to 2 decimal places."""
        for col in cols:
            frame[col] = frame[col].round(decimals=2)

    def lower(frame, cols):
        """Lowercase columns.

        Args:
            frame: pandas.DataFrame
            cols: List, columns to lowercase, must exist in frame

        Returns:
            None
        """
        for col in cols:
            frame[col] = frame[col].str.lower()

    def zhang2021ignorance(frame):
        """Scale fairness metrics according to zhang2021ignorance.

        Args:
            frame: pandas.Dataframe

        Returns:
            None

        Note:
            This method scales the fairness metrics according to the
            techniques outlined in the zhang2021ignorance paper. The
            absolute values for all fairness metrics are used. For
            disparate impact, the distance to 1 is calculated and then the
            values are scaled between [0, 1]. This means that all fairness
            metrics are between [0, 1] & higher values indicate more
            unfairness.
        """
        scaler = MinMaxScaler()
        frame["disparate_impact"] = frame["disparate_impact"] - 1
        frame[METRICS] = frame[METRICS].abs()
        frame["disparate_impact"] = scaler.fit_transform(
            frame["disparate_impact"].values.reshape(-1, 1)
        ).ravel()

    try:
        lower(frame=frame, cols=["protected"])
    except KeyError:
        pass

    try:
        round(frame=frame, cols=["frac"])
    except KeyError:
        pass
    
    zhang2021ignorance(frame=frame)
