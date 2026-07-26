from pathlib import Path
import math

import pandas as pd
import pytest

from src.data_ingestion.canonical_builder import build_canonical_dataset
from src.data_ingestion.csv_reader import CsvReadResult
from src.feature_engine import FeatureDataset, extract_features


def test_canonical_dataset_to_feature_dataset_pipeline() -> None:
    reader_result = CsvReadResult(
        source_file="BL011.csv",
        absolute_path=str(Path("BL011.csv").resolve()),
        source_type="csv",
        encoding="utf-8",
        delimiter=",",
        strain_label_original="BL011",
        row_count=3,
        column_count=7,
        original_columns=[
            "bacteria_id",
            "antibiotic",
            "concentration",
            "Experiment",
            "replicate",
            "time_min",
            "luminescence",
        ],
        dataframe=pd.DataFrame(
            {
                "bacteria_id": ["BL011", "BL011", "BL011"],
                "antibiotic": ["Diazinon", "Diazinon", "Diazinon"],
                "concentration": ["5", "5", "5"],
                "Experiment": ["1", "1", "1"],
                "replicate": ["1", "1", "1"],
                "time_min": ["0", "5", "10"],
                "luminescence": ["10", "20", "15"],
            }
        ),
        warnings=[],
    )
    canonical_result = build_canonical_dataset([reader_result])

    feature_dataset = extract_features(canonical_result.dataframe)

    assert isinstance(feature_dataset, FeatureDataset)
    assert len(feature_dataset.dataframe) == 1
    row = feature_dataset.dataframe.iloc[0]
    assert row["Experiment_ID"] == "csv_BL011_experiment_1"
    assert row["Source_File"] == "BL011.csv"
    assert row["Measurement_Unit_ID"] == "unit_r000001__col007_luminescence"
    assert row["Strain"] == "BL011"
    assert row["Chemical"] == "Diazinon"
    assert row["Concentration"] == "5"
    assert row["baseline"] == 10.0
    assert row["peak"] == 20.0
    assert row["endpoint"] == 15.0
    assert row["auc"] == 162.5
    assert row["initial_slope"] == 2.0
    assert row["maximum_slope"] == 2.0
    assert row["fold_change"] == 1.0
    assert row["log2_fold_change"] == pytest.approx(math.log2(1.5))
    assert feature_dataset.metadata["raw_readers_used"] is False
    assert feature_dataset.summary["feature_rows"] == 1
    assert feature_dataset.qc.passed is True

