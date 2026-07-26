import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import openpyxl
import pandas as pd

from src.data_ingestion.canonical_builder import build_canonical_dataset
from src.data_ingestion.csv_reader import read_biosensor_csv
from src.data_ingestion.excel_reader import read_biosensor_excel
from src.data_schema.canonical_schema import CANONICAL_COLUMNS
from src.quality_control.canonical_qc import audit_canonical_dataframe


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


def test_reader_builder_qc_keeps_same_condition_different_wells_separate() -> None:
    with local_test_workspace("identity_different_wells") as workspace:
        csv_path = workspace / "BL027ab.csv"
        csv_path.write_text(
            "bacteria_id,antibiotic,concentration,Experiment,replicate,Well_ID,time_min,luminescence\n"
            "BL027,Lambda Cyclotherin,5,1,1,A01,0,100\n"
            "BL027,Lambda Cyclotherin,5,1,1,A02,0,101\n",
            encoding="utf-8",
        )
        before = (csv_path.read_bytes(), csv_path.stat().st_mtime_ns, csv_path.stat().st_size)

        reader_result = read_biosensor_csv(csv_path)
        input_before = reader_result.dataframe.copy(deep=True)
        build_result = build_canonical_dataset([reader_result])
        qc_result = audit_canonical_dataframe(build_result.dataframe)

        assert build_result.row_count == 2
        assert build_result.schema_valid is True
        assert build_result.dataframe["Measurement_Unit_ID"].tolist() == ["well_A01", "well_A02"]
        assert qc_result.logical_duplicate_count == 0
        assert qc_result.row_count == 2
        assert set(build_result.dataframe["Chemical_Name_Original"]) == {"Lambda Cyclotherin"}
        assert set(build_result.dataframe["Strain_Original"]) == {"BL027ab"}
        pd.testing.assert_frame_equal(reader_result.dataframe, input_before)
        assert (csv_path.read_bytes(), csv_path.stat().st_mtime_ns, csv_path.stat().st_size) == before


def test_duplicate_csv_measurement_columns_are_source_position_units() -> None:
    with local_test_workspace("identity_duplicate_csv_columns") as workspace:
        csv_path = workspace / "BL011.csv"
        csv_path.write_text(
            "bacteria_id,antibiotic,concentration,Experiment,replicate,time_min,luminescence,luminescence\n"
            "BL011,Diazinon,5,1,1,0,100,101\n",
            encoding="utf-8",
        )

        build_result = build_canonical_dataset([read_biosensor_csv(csv_path)])
        qc_result = audit_canonical_dataframe(build_result.dataframe)

        assert build_result.row_count == 2
        assert build_result.dataframe["Source_Row_ID"].tolist() == [1, 1]
        assert build_result.dataframe["Measurement_Unit_ID"].nunique() == 2
        assert build_result.dataframe["Measurement_Unit_ID"].str.contains("__col").all()
        assert build_result.dataframe["QC_Flags"].str.contains("measurement_unit_id_uses_source_column").all()
        assert qc_result.logical_duplicate_count == 0
        assert qc_result.legacy_logical_duplicate_count == 2
        assert qc_result.separate_replicate_measurement_count == 2


def test_repeated_timepoint_within_one_unit_is_detected() -> None:
    with local_test_workspace("identity_repeated_timepoint") as workspace:
        csv_path = workspace / "BL011.csv"
        csv_path.write_text(
            "bacteria_id,antibiotic,concentration,Experiment,replicate,time_min,luminescence\n"
            "BL011,Diazinon,5,1,1,0,100\n"
            "BL011,Diazinon,5,1,1,5,105\n"
            "BL011,Diazinon,5,1,1,5,106\n",
            encoding="utf-8",
        )

        build_result = build_canonical_dataset([read_biosensor_csv(csv_path)])
        qc_result = audit_canonical_dataframe(build_result.dataframe)

        assert build_result.row_count == 3
        assert build_result.dataframe["Measurement_Unit_ID"].nunique() == 1
        assert qc_result.logical_duplicate_count == 2
        assert qc_result.conflicting_value_duplicate_count == 2
        assert qc_result.duplicate_timepoint_group_count == 1


def test_different_replicates_same_time_are_not_duplicates() -> None:
    with local_test_workspace("identity_different_replicates") as workspace:
        csv_path = workspace / "BL011.csv"
        csv_path.write_text(
            "bacteria_id,antibiotic,concentration,Experiment,replicate,time_min,luminescence\n"
            "BL011,Diazinon,5,1,1,0,100\n"
            "BL011,Diazinon,5,1,2,0,101\n",
            encoding="utf-8",
        )

        build_result = build_canonical_dataset([read_biosensor_csv(csv_path)])
        qc_result = audit_canonical_dataframe(build_result.dataframe)

        assert build_result.row_count == 2
        assert build_result.dataframe["Measurement_Unit_ID"].nunique() == 2
        assert qc_result.logical_duplicate_count == 0
        assert qc_result.legacy_logical_duplicate_count == 0


def test_measurement_unit_ids_are_deterministic_and_not_signal_based() -> None:
    with local_test_workspace("identity_deterministic") as workspace:
        csv_path = workspace / "BL011.csv"
        changed_signal_path = workspace / "BL011_changed.csv"
        csv_path.write_text(
            "bacteria_id,antibiotic,concentration,Experiment,replicate,time_min,luminescence\n"
            "BL011,Diazinon,5,1,1,0,100\n"
            "BL011,Diazinon,5,1,1,5,105\n",
            encoding="utf-8",
        )
        changed_signal_path.write_text(
            "bacteria_id,antibiotic,concentration,Experiment,replicate,time_min,luminescence\n"
            "BL011,Diazinon,5,1,1,0,999\n"
            "BL011,Diazinon,5,1,1,5,1000\n",
            encoding="utf-8",
        )

        first = build_canonical_dataset([read_biosensor_csv(csv_path)])
        second = build_canonical_dataset([read_biosensor_csv(csv_path)])
        changed_signal = build_canonical_dataset([read_biosensor_csv(changed_signal_path)])

        assert first.dataframe["Measurement_Unit_ID"].tolist() == second.dataframe["Measurement_Unit_ID"].tolist()
        assert first.dataframe["Measurement_Unit_ID"].tolist() == changed_signal.dataframe["Measurement_Unit_ID"].tolist()
        assert first.dataframe["Source_Row_ID"].tolist() == second.dataframe["Source_Row_ID"].tolist()
        assert first.dataframe["QC_Flags"].str.contains("measurement_unit_id_synthetic").all()


def test_excel_worksheet_identity_is_retained() -> None:
    with local_test_workspace("identity_excel_worksheet") as workspace:
        excel_path = workspace / "BL011.12hrs.xlsx"
        workbook = openpyxl.Workbook()
        worksheet = workbook.active
        worksheet.title = "RunA"
        worksheet.append(["", "bacteria_id", "antibiotic", "concentration", "Experiment", "replicate", "time_min", "luminescence"])
        worksheet.append([None, "BL011", "Lambda Cyclotherin", 5, 1, 1, 0, 100])
        workbook.save(excel_path)
        workbook.close()

        build_result = build_canonical_dataset([read_biosensor_excel(excel_path)])

        assert build_result.row_count == 1
        assert build_result.schema_valid is True
        assert build_result.dataframe.loc[0, "Worksheet"] == "RunA"
        assert build_result.dataframe.loc[0, "Chemical_Name_Original"] == "Lambda Cyclotherin"


def test_missing_measurement_unit_identity_is_ambiguous_in_qc() -> None:
    row = {
        "Experiment_ID": "EXP-1",
        "Plate_ID": pd.NA,
        "Source_File": "BL011.csv",
        "Source_Path": pd.NA,
        "Source_Type": "csv",
        "Worksheet": pd.NA,
        "Data_Source": "24_hour_csv",
        "Time_Series_Duration_Hours": 24.0,
        "Analysis_Window": "unassigned",
        "Import_Timestamp": pd.NaT,
        "Source_Row_ID": 1,
        "Measurement_Unit_ID": pd.NA,
        "Strain_Original": "BL011",
        "Strain_Standardized": pd.NA,
        "Chemical_Name_Original": "Diazinon",
        "Chemical_Name_Standardized": pd.NA,
        "Concentration_Label": "5",
        "Concentration_ug_mL": 5.0,
        "Control_Status": "treatment",
        "Control_Type": pd.NA,
        "Replicate_ID": pd.NA,
        "Replicate_Type": "unspecified",
        "Well_ID": pd.NA,
        "Time_Original": "0",
        "Time_Unit_Original": "min",
        "Time_Minutes": 0.0,
        "Time_Hours": 0.0,
        "Timepoint_Index": 0,
        "Luminescence_Raw": 100.0,
        "Luminescence_Normalized": pd.NA,
        "Normalization_Method": pd.NA,
        "QC_Status": "warning",
        "QC_Flags": "measurement_unit_identity_ambiguous",
        "Record_Valid": True,
        "Notes": pd.NA,
    }
    dataframe = pd.DataFrame([row], columns=list(CANONICAL_COLUMNS))

    result = audit_canonical_dataframe(dataframe)

    assert result.unresolved_measurement_unit_count == 1
    assert result.ambiguous_measurement_identity_count == 1
