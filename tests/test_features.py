import pandas as pd
from src.features.build import build_features


def test_build_features_returns_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = build_features(df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape
