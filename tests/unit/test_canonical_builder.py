import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest

from src.data_ingestion.canonical_builder import (
    build_canonical_dataset,
    build_canonical_from_csv,
    build_canonical_from_excel,
    save_canonical_dataset,
)
from src.data_ingestion.csv_reader import CsvReadResult
from src.data_ingestion.excel_reader import ExcelReadResult
from src.data_schema.canonical_schema import CANONICAL_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_TMP_ROOT = PROJECT_ROOT / "tests" / "tmp"


@contextmanager
def local_test_workspace(test_name: str) -> Iterator[Path]:
    workspace = TEST_TMP_ROOT / test_name
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    try:
        yield workspace
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)
        if TEST_TMP_ROOT.exists() and not any(TEST_TMP_ROOT.iterdir()):
            TEST_TMP_ROOT.rmdir()


def csv_reader_result(
    path: Path,
    dataframe: pd.DataFrame | None = None,
    strain: str = "BL011",
    warnings: list[str] | None = None,
) -> CsvReadResult:
    dataframe = dataframe if dataframe is not None else pd.DataFrame(
        {
            "bacteria_id": ["BL011"],
            "antibiotic": ["Diazinon"],
            "concentration": ["5"],
            "Experiment": ["1"],
            "replicate": ["1"],
            "time_min": ["60"],
            "luminescence": ["123.5"],
        }
    )
    return CsvReadResult(
        source_file=path.name,
        absolute_path=str(path.resolve()),
        source_type="csv_24h_candidate",
        encoding="utf-8",
        delimiter=",",
        strain_label_original=strain,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        original_columns=list(dataframe.columns),
        dataframe=dataframe,
        warnings=warnings or [],
    )


def excel_reader_result(
    path: Path,
    dataframe: pd.DataFrame | None = None,
    strain: str = "BL011",
    warnings: list[str] | None = None,
) -> ExcelReadResult:
    dataframe = dataframe if dataframe is not None else pd.DataFrame(
        {
            "": [None],
            "bacteria_id": ["BL011"],
            "antibiotic": ["Lambda Cyclotherin"],
            "concentration": [5],
            "Experiment": [1],
            "replicate": [1],
            "time_min": [60],
            "luminescence": [123.5],
        }
    )
    return ExcelReadResult(
        filename=path.name,
        workbook_name=path.name,
        absolute_path=str(path.resolve()),
        worksheet_names=["Sheet1"],
        active_worksheet="Sheet1",
        inferred_strain=strain,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        original_column_names=list(dataframe.columns),
        dataframe=dataframe,
        warnings=warnings or [],
    )


def test_one_valid_csv_style_reader_result_produces_canonical_rows() -> None:
    with local_test_workspace("builder_csv") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv"))

        assert result.row_count == 1
        assert result.dataframe.loc[0, "Source_Type"] == "csv"
        assert result.dataframe.loc[0, "Data_Source"] == "24_hour_csv"


def test_one_valid_excel_style_reader_result_produces_canonical_rows() -> None:
    with local_test_workspace("builder_excel") as workspace:
        result = build_canonical_from_excel(excel_reader_result(workspace / "BL011.12hrs.xlsx"))

        assert result.row_count == 1
        assert result.dataframe.loc[0, "Source_Type"] == "xlsx"
        assert result.dataframe.loc[0, "Data_Source"] == "12_hour_excel"
        assert result.dataframe.loc[0, "Worksheet"] == "Sheet1"


def test_mixed_csv_and_excel_results_combine_correctly() -> None:
    with local_test_workspace("builder_mixed") as workspace:
        result = build_canonical_dataset(
            [
                csv_reader_result(workspace / "BL011.csv"),
                excel_reader_result(workspace / "BL011.12hrs.xlsx"),
            ]
        )

        assert result.row_count == 2
        assert result.source_files == ["BL011.csv", "BL011.12hrs.xlsx"]


def test_canonical_column_order_is_exact() -> None:
    with local_test_workspace("builder_order") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv"))

        assert list(result.dataframe.columns) == list(CANONICAL_COLUMNS)


def test_original_labels_are_preserved() -> None:
    dataframe = pd.DataFrame(
        {
            "bacteria_id": ["BL011"],
            "antibiotic": ["Raw Label"],
            "concentration": ["0.05"],
            "Experiment": ["1"],
            "replicate": ["1"],
            "time_min": ["0"],
            "luminescence": ["1"],
        }
    )
    with local_test_workspace("builder_original_labels") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv", dataframe=dataframe))

        assert result.dataframe.loc[0, "Chemical_Name_Original"] == "Raw Label"
        assert result.dataframe.loc[0, "Strain_Original"] == "BL011"


def test_lambda_cyclotherin_remains_unchanged() -> None:
    with local_test_workspace("builder_lambda") as workspace:
        result = build_canonical_from_excel(excel_reader_result(workspace / "BL011.12hrs.xlsx"))

        assert result.dataframe.loc[0, "Chemical_Name_Original"] == "Lambda Cyclotherin"


def test_bl027ab_remains_unchanged() -> None:
    with local_test_workspace("builder_bl027ab") as workspace:
        result = build_canonical_from_csv(
            csv_reader_result(workspace / "BL027ab.csv", strain="BL027ab")
        )

        assert result.dataframe.loc[0, "Strain_Original"] == "BL027ab"


def test_luminescence_raw_is_preserved() -> None:
    with local_test_workspace("builder_lum_raw") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv"))

        assert result.dataframe.loc[0, "Luminescence_Raw"] == 123.5


def test_luminescence_normalized_remains_null() -> None:
    with local_test_workspace("builder_lum_norm") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv"))

        assert pd.isna(result.dataframe.loc[0, "Luminescence_Normalized"])


def test_analysis_window_defaults_to_unassigned() -> None:
    with local_test_workspace("builder_window") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv"))

        assert result.dataframe.loc[0, "Analysis_Window"] == "unassigned"


def test_time_units_are_converted_consistently() -> None:
    with local_test_workspace("builder_time") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv"))

        assert result.dataframe.loc[0, "Time_Original"] == "60"
        assert result.dataframe.loc[0, "Time_Unit_Original"] == "min"
        assert result.dataframe.loc[0, "Time_Minutes"] == 60
        assert result.dataframe.loc[0, "Time_Hours"] == 1


def test_concentration_labels_are_preserved() -> None:
    with local_test_workspace("builder_conc_label") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv"))

        assert result.dataframe.loc[0, "Concentration_Label"] == "5"


def test_numeric_concentration_is_parsed() -> None:
    with local_test_workspace("builder_conc_numeric") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv"))

        assert result.dataframe.loc[0, "Concentration_ug_mL"] == 5


def test_controls_can_have_null_numeric_concentration() -> None:
    dataframe = pd.DataFrame(
        {
            "bacteria_id": ["BL011"],
            "antibiotic": ["Diazinon"],
            "concentration": ["Control"],
            "Experiment": ["1"],
            "replicate": ["1"],
            "time_min": ["60"],
            "luminescence": ["123.5"],
        }
    )
    with local_test_workspace("builder_control") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv", dataframe=dataframe))

        assert pd.isna(result.dataframe.loc[0, "Concentration_ug_mL"])
        assert result.dataframe.loc[0, "Control_Status"] == "control"


def test_missing_optional_metadata_does_not_crash_builder() -> None:
    with local_test_workspace("builder_optional_missing") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv"))

        assert pd.isna(result.dataframe.loc[0, "Plate_ID"])
        assert pd.isna(result.dataframe.loc[0, "Well_ID"])
        assert result.row_count == 1


def test_missing_replicate_metadata_produces_ambiguous_warning() -> None:
    dataframe = pd.DataFrame(
        {
            "bacteria_id": ["BL011"],
            "antibiotic": ["Diazinon"],
            "concentration": ["5"],
            "Experiment": ["1"],
            "time_min": ["60"],
            "luminescence": ["123.5"],
        }
    )
    with local_test_workspace("builder_required_missing") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv", dataframe=dataframe))

        assert result.schema_valid is True
        assert result.warning_record_count == 1
        assert "missing_Replicate_ID" in result.dataframe.loc[0, "QC_Flags"]
        assert "measurement_unit_identity_ambiguous" in result.dataframe.loc[0, "QC_Flags"]


def test_blank_replicate_metadata_produces_ambiguous_warning() -> None:
    dataframe = pd.DataFrame(
        {
            "bacteria_id": ["BL011"],
            "antibiotic": ["Diazinon"],
            "concentration": ["5"],
            "Experiment": ["1"],
            "replicate": [""],
            "time_min": ["60"],
            "luminescence": ["123.5"],
        }
    )
    with local_test_workspace("builder_blank_required") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv", dataframe=dataframe))

        assert result.schema_valid is True
        assert result.warning_record_count == 1
        assert pd.isna(result.dataframe.loc[0, "Replicate_ID"])
        assert "missing_Replicate_ID" in result.dataframe.loc[0, "QC_Flags"]


def test_duplicate_measurement_keys_are_flagged() -> None:
    dataframe = pd.DataFrame(
        {
            "bacteria_id": ["BL011", "BL011"],
            "antibiotic": ["Diazinon", "Diazinon"],
            "concentration": ["5", "5"],
            "Experiment": ["1", "1"],
            "replicate": ["1", "1"],
            "time_min": ["60", "60"],
            "luminescence": ["123.5", "124.5"],
        }
    )
    with local_test_workspace("builder_duplicate") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv", dataframe=dataframe))

        assert "duplicate_logical_records" in result.schema_validation.row_problem_counts


def test_same_condition_in_different_wells_is_not_a_duplicate_measurement_key() -> None:
    dataframe = pd.DataFrame(
        {
            "bacteria_id": ["BL011", "BL011"],
            "antibiotic": ["Diazinon", "Diazinon"],
            "concentration": ["5", "5"],
            "Experiment": ["1", "1"],
            "replicate": ["1", "1"],
            "Well_ID": ["A01", "A02"],
            "time_min": ["60", "60"],
            "luminescence": ["123.5", "124.5"],
        }
    )
    with local_test_workspace("builder_wells_distinct") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv", dataframe=dataframe))

        assert result.dataframe["Measurement_Unit_ID"].tolist() == ["well_A01", "well_A02"]
        assert "duplicate_logical_records" not in result.schema_validation.row_problem_counts


def test_duplicate_luminescence_column_names_are_expanded_as_distinct_units() -> None:
    dataframe = pd.DataFrame(
        [["BL011", "Diazinon", "5", "1", "1", "60", "123.5", "124.5"]],
        columns=[
            "bacteria_id",
            "antibiotic",
            "concentration",
            "Experiment",
            "replicate",
            "time_min",
            "luminescence",
            "luminescence",
        ],
    )
    with local_test_workspace("builder_duplicate_lum_columns") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv", dataframe=dataframe))

        assert result.row_count == 2
        assert result.dataframe["Source_Row_ID"].tolist() == [1, 1]
        assert result.dataframe["Luminescence_Raw"].tolist() == [123.5, 124.5]
        assert result.dataframe["Measurement_Unit_ID"].nunique() == 2
        assert all("__col" in unit_id for unit_id in result.dataframe["Measurement_Unit_ID"])
        assert result.dataframe["QC_Flags"].str.contains("measurement_unit_id_uses_source_column").all()


def test_measurement_unit_id_is_deterministic_and_signal_independent() -> None:
    first_dataframe = pd.DataFrame(
        {
            "bacteria_id": ["BL011", "BL011"],
            "antibiotic": ["Diazinon", "Diazinon"],
            "concentration": ["5", "5"],
            "Experiment": ["1", "1"],
            "replicate": ["1", "1"],
            "time_min": ["0", "5"],
            "luminescence": ["100", "101"],
        }
    )
    second_dataframe = first_dataframe.copy(deep=True)
    second_dataframe["luminescence"] = ["999", "1000"]

    with local_test_workspace("builder_unit_id_deterministic") as workspace:
        first = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv", dataframe=first_dataframe))
        second = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv", dataframe=second_dataframe))

        assert first.dataframe["Measurement_Unit_ID"].tolist() == second.dataframe["Measurement_Unit_ID"].tolist()
        assert first.dataframe["Measurement_Unit_ID"].nunique() == 1
        assert first.dataframe.loc[0, "Measurement_Unit_ID"].startswith("unit_r000001")


def test_negative_luminescence_is_retained_and_warned() -> None:
    dataframe = pd.DataFrame(
        {
            "bacteria_id": ["BL011"],
            "antibiotic": ["Diazinon"],
            "concentration": ["5"],
            "Experiment": ["1"],
            "replicate": ["1"],
            "time_min": ["60"],
            "luminescence": ["-1"],
        }
    )
    with local_test_workspace("builder_negative_lum") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv", dataframe=dataframe))

        assert result.dataframe.loc[0, "Luminescence_Raw"] == -1
        assert result.dataframe.loc[0, "QC_Status"] == "warning"
        assert "negative_luminescence_raw" in result.dataframe.loc[0, "QC_Flags"]


def test_infinite_luminescence_is_invalid() -> None:
    dataframe = pd.DataFrame(
        {
            "bacteria_id": ["BL011"],
            "antibiotic": ["Diazinon"],
            "concentration": ["5"],
            "Experiment": ["1"],
            "replicate": ["1"],
            "time_min": ["60"],
            "luminescence": ["inf"],
        }
    )
    with local_test_workspace("builder_inf_lum") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv", dataframe=dataframe))

        assert result.invalid_record_count == 1
        assert "infinite_luminescence_raw" in result.dataframe.loc[0, "QC_Flags"]


def test_source_row_id_is_deterministic() -> None:
    with local_test_workspace("builder_row_id") as workspace:
        first = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv"))
        second = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv"))

        assert first.dataframe["Source_Row_ID"].tolist() == second.dataframe["Source_Row_ID"].tolist()


def test_experiment_id_is_deterministic() -> None:
    with local_test_workspace("builder_exp_id") as workspace:
        first = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv"))
        second = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv"))

        assert first.dataframe["Experiment_ID"].tolist() == second.dataframe["Experiment_ID"].tolist()


def test_input_reader_dataframe_is_not_mutated() -> None:
    with local_test_workspace("builder_no_mutation") as workspace:
        reader_result = csv_reader_result(workspace / "BL011.csv")
        before = reader_result.dataframe.copy(deep=True)

        build_canonical_from_csv(reader_result)

        pd.testing.assert_frame_equal(reader_result.dataframe, before)


def test_output_validation_is_performed() -> None:
    with local_test_workspace("builder_validation") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv"))

        assert result.schema_validation is not None
        assert result.schema_valid is True


def test_output_saving_refuses_to_overwrite_by_default() -> None:
    with local_test_workspace("builder_save_refuse") as workspace:
        result = build_canonical_from_csv(csv_reader_result(workspace / "BL011.csv"))
        output_path = workspace / "canonical.csv"
        output_path.write_text("existing", encoding="utf-8")

        with pytest.raises(FileExistsError):
            save_canonical_dataset(result, output_path)


def test_source_files_are_never_modified() -> None:
    with local_test_workspace("builder_source_safety") as workspace:
        source_path = workspace / "BL011.csv"
        source_path.write_text("raw source placeholder", encoding="utf-8")
        before = (source_path.read_bytes(), source_path.stat().st_mtime_ns, source_path.stat().st_size)

        build_canonical_from_csv(csv_reader_result(source_path))

        after = (source_path.read_bytes(), source_path.stat().st_mtime_ns, source_path.stat().st_size)
        assert after == before
