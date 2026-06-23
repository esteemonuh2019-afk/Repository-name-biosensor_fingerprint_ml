import pandas as pd

from src.model_training.models import (
    NUMERIC_FEATURE_COLUMNS,
    predict_classifier,
    predict_regressor,
    train_classifier,
    train_regressor,
)


def test_classifier_trains_successfully() -> None:
    model, feature_columns = train_classifier(_feature_dataframe())

    assert hasattr(model, "predict")
    assert feature_columns == list(NUMERIC_FEATURE_COLUMNS)


def test_classifier_predicts_expected_number_of_labels() -> None:
    feature_df = _feature_dataframe()
    model, feature_columns = train_classifier(feature_df)

    predictions = predict_classifier(model, feature_df, feature_columns)

    assert len(predictions) == len(feature_df)


def test_regressor_trains_successfully() -> None:
    model, feature_columns = train_regressor(_feature_dataframe())

    assert hasattr(model, "predict")
    assert feature_columns == list(NUMERIC_FEATURE_COLUMNS)


def test_regressor_predicts_expected_number_of_values() -> None:
    feature_df = _feature_dataframe()
    model, feature_columns = train_regressor(feature_df)

    predictions = predict_regressor(model, feature_df, feature_columns)

    assert len(predictions) == len(feature_df)


def test_feature_column_list_is_preserved() -> None:
    classifier_model, classifier_columns = train_classifier(_feature_dataframe())
    regressor_model, regressor_columns = train_regressor(_feature_dataframe())

    assert classifier_columns == list(NUMERIC_FEATURE_COLUMNS)
    assert regressor_columns == list(NUMERIC_FEATURE_COLUMNS)
    assert len(predict_classifier(classifier_model, _feature_dataframe(), classifier_columns)) == 6
    assert len(predict_regressor(regressor_model, _feature_dataframe(), regressor_columns)) == 6


def _feature_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chemical": [
                "Diazinon",
                "Diazinon",
                "DEET",
                "DEET",
                "Propoxur",
                "Propoxur",
            ],
            "concentration": [5.0, 50.0, 5.0, 50.0, 5.0, 50.0],
            "auc": [6125.0, 8200.0, 5362.5, 7100.0, 5900.0, 7800.0],
            "max_signal": [1250.0, 1500.0, 1100.0, 1350.0, 1180.0, 1420.0],
            "min_signal": [1005.0, 1002.0, 990.0, 988.0, 975.0, 970.0],
            "time_to_peak": [5.0, 5.0, 5.0, 10.0, 10.0, 10.0],
            "initial_slope": [49.0, 79.6, 22.0, 39.4, 41.0, 68.0],
            "final_signal": [1180.0, 1400.0, 1080.0, 1185.0, 1160.0, 1390.0],
        }
    )
