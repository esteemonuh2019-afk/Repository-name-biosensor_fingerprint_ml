# Phase 0 Safety and Environment Audit

## 1. Audit Date and Time

- Audit timestamp: 2026-07-25 20:35:56 +03:00
- Audit phase: Phase 0 only
- Scope: Read-only project safety and preparation audit, plus this single permitted report file.

## 2. Project Root

- Absolute project path: `C:\Users\USER\Desktop\biosensor_fingerprint_ml`
- Root folder name: `biosensor_fingerprint_ml`
- Result: Correct project root confirmed.

## 3. Workspace Confirmation

- The active Codex workspace root is `C:\Users\USER\Desktop\biosensor_fingerprint_ml`.
- The runtime context exposes one workspace root.
- No evidence was found that another workspace folder is open in this Codex session.
- No `.vscode` directory was present, so VS Code workspace settings were not directly inspectable from project files.

## 4. Backup Confirmation

- User-confirmed backup: Yes. The user stated that a separate backup already exists.
- External backup inspection: Not performed, because no external backup location was documented in the project.
- Obvious backup/copy folders inside the project: None found in the project tree outside ignored environment/cache directories.
- No duplicate or backup files were deleted or modified.

## 5. Python Environment Findings

- `.venv` exists: Yes, at `C:\Users\USER\Desktop\biosensor_fingerprint_ml\.venv`.
- `.venv\Scripts\python.exe` exists and reports: Python 3.14.4.
- Current shell `python` resolves to: `C:\Program Files\Python314\python.exe`.
- Current shell `python --version`: Python 3.14.4.
- `VIRTUAL_ENV` in the current shell: Not set.
- `.venv\pyvenv.cfg` reports:
  - `home = C:\Program Files\Python314`
  - `version = 3.14.4`
  - `include-system-site-packages = false`
- VS Code selected interpreter: Not detectable from project files because `.vscode/settings.json` is absent.
- Selected interpreter belongs to project `.venv`: Not confirmable from VS Code settings. The current shell interpreter is global, not the `.venv` path.

## 6. Dependency-File Findings

- Present:
  - `requirements.txt`
- Missing:
  - `requirements-dev.txt`
  - `pyproject.toml`
  - `setup.py`
  - `setup.cfg`
  - `Pipfile`
  - `poetry.lock`
  - `environment.yml`
- `requirements.txt` exists but dependencies are unpinned.
- No packages were installed, upgraded, removed, or imported for a smoke test during this audit.

## 7. Git Findings

- Valid Git repository: Yes.
- Branch: `main`
- Upstream shown by status: `origin/main`
- Remote configured: Yes.
  - `origin` fetch/push: `https://github.com/esteemonuh2019-afk/Repository-name-biosensor_fingerprint_ml.git`
- Latest commit:
  - SHA: `b03241001789fe8dacd744305ad8b7af7ef4cfea`
  - Date: `2026-06-24 13:11:52 +0300`
  - Subject: `Add sample size justification and robustness evidence`
- Initial Git status before this report was created:
  - Modified: `.gitignore`
  - Untracked:
    - `scripts/run_confidence_intervals.py`
    - `scripts/run_repeated_runs.py`
    - `src/model_evaluation/confidence_intervals.py`
    - `src/model_evaluation/repeated_runs.py`
    - `tests/unit/test_confidence_intervals.py`
    - `tests/unit/test_repeated_runs.py`
- Staged files: None found.
- Deleted or renamed files: None found in `git status --short`.
- `.gitignore` exists: Yes.
- `.venv` ignored: Yes, via `.gitignore:1:.venv/`.
- `.venv` tracked by Git: No.
- `data/raw` tracked by Git: No.
- `data/raw` ignored by Git: Yes, via `.gitignore`.
- Generated outputs tracked by Git: Yes. `git ls-files outputs` reports 40 tracked output files.
- Large ignored generated output present: `outputs/tables/cleaned_data.csv`, about 118 MB, ignored by `.gitignore`.

## 8. Top-Level Structure

- `.agents/`: Local agent metadata or support files.
- `.github/`: GitHub configuration.
- `.pytest_cache/`: Pytest cache, ignored/generated.
- `.venv/`: Project virtual environment, ignored.
- `data/`: Data area.
- `data/raw/`: Raw experimental CSV data.
- `docs/`: Project documentation, audit and validation documents.
- `outputs/`: Generated figures, reports, metrics, and tables.
- `outputs/figures/`: Generated PNG figures.
- `outputs/reports/`: Generated Markdown reports.
- `outputs/tables/`: Generated CSV and JSON tables.
- `scripts/`: Executable workflow scripts.
- `src/`: Python implementation modules.
- `tests/`: Unit, integration, regression, black-box, contract, and end-to-end tests.
- Root files:
  - `.gitignore`
  - `README.md`
  - `requirements.txt`

Likely locations by category:

- Raw data: `data/raw/`
- Processed data: `outputs/tables/processed_data.csv`, `outputs/tables/cleaned_data.csv`
- Scripts: `scripts/`
- Python packages/modules: `src/`
- Tests: `tests/`
- GUI code: None found.
- Reports: `outputs/reports/`, `docs/`
- Figures/heatmaps: `outputs/figures/`
- Models: No `.pkl`, `.joblib`, `.sav`, or `.model` files found. `outputs/models/` is ignored but absent.
- Logs: No top-level log directory found. Tests reference `outputs/logs/unit.log`, but no such directory was found.
- Configuration files: `.gitignore`, `requirements.txt`, `.github/`
- Documentation: `README.md`, `docs/`

## 9. Likely Entry Points

- `scripts/run_real_analysis.py`
  - Has `if __name__ == "__main__"`.
  - Reads `data/raw/*.csv`.
  - Runs the full analysis, model training/evaluation when possible, figure generation, and report/table writing to `outputs/`.
  - This appears to be the main real-data workflow script.
- `src/pipeline/run_pipeline.py`
  - Defines `run_analysis_pipeline(input_files, output_dir)`.
  - Imported by tests and `scripts/run_real_analysis.py`.
  - Does not expose a command-line `__main__`, but is a central callable pipeline entry point.
- `src/data_validation/inspect_dataset.py`
  - Uses `argparse`.
  - Has `if __name__ == "__main__"`.
  - Defaults to inspecting `data/raw/`.
  - Reads and prints summaries without writing files.
- Other executable workflow scripts with `if __name__ == "__main__"`:
  - `scripts/analyze_feature_importance.py`
  - `scripts/run_advanced_feature_generation.py`
  - `scripts/run_advanced_loeo_comparison.py`
  - `scripts/run_advanced_per_chemical_analysis.py`
  - `scripts/run_chemical_specific_strains.py`
  - `scripts/run_confidence_intervals.py`
  - `scripts/run_loeo_validation.py`
  - `scripts/run_normalized_loeo.py`
  - `scripts/run_panel_optimization.py`
  - `scripts/run_per_chemical_analysis.py`
  - `scripts/run_repeated_runs.py`
  - `scripts/run_specialist_ensemble.py`
  - `scripts/run_strain_ablation.py`
- No `main.py`, `app.py`, `biosensor_gui.py`, `__main__.py`, batch file, PowerShell launcher, shell script, desktop launcher, Streamlit app, Tkinter app, or PyQt app was found in the targeted entry-point scan.
- README guidance says to run workflow scripts from `scripts/`, but it does not identify one primary command.

## 10. Raw-Data Locations

Primary raw experimental data:

- Path: `data/raw/`
- File types: `.csv`
- Approximate file count: 6
- Approximate total size: 68,952,446 bytes
- Git tracking: Not tracked.
- Git ignore status: Ignored by `.gitignore`.
- Files found:
  - `BL011.csv` - 12,210,942 bytes
  - `BL027ab.csv` - 12,129,014 bytes
  - `BL029.csv` - 10,073,474 bytes
  - `BL030.csv` - 11,266,710 bytes
  - `BL031.csv` - 11,068,389 bytes
  - `BL032.csv` - 12,203,917 bytes

Test fixture data:

- Path: `tests/fixtures/`
- File types: `.csv`, `.json`
- Approximate file count: 9
- Purpose: Test fixtures, not primary raw experimental data.

External 12-hour Excel datasets:

- Location: Outside the project folder per user context.
- Audit action: Not copied, inspected, imported, or modified.

## 11. Raw-Data Overwrite-Risk Assessment

- No evidence was found that source or script code writes to `data/raw/`.
- No evidence was found of deleting, renaming, moving, or clearing raw-data directories in `src/` or `scripts/`.
- No `inplace=True` pattern was found in the targeted overwrite-risk search.
- `src/data_ingestion/loader.py` reads CSV files with `pd.read_csv` and returns DataFrames.
- `src/pipeline/run_pipeline.py` reads input files, transforms data in memory, and writes a Markdown report to the supplied output directory.
- `scripts/run_real_analysis.py` reads `data/raw/*.csv` and writes generated artifacts to fixed paths under `outputs/`.
- `src/data_validation/inspect_dataset.py` reads CSV files and prints structural summaries; no writes were found.
- Tests contain temporary-workspace deletion patterns using `shutil.rmtree`, but the inspected test code targets test workspaces, not `data/raw/`.
- Main overwrite risk identified: generated outputs under `outputs/` are overwritten by fixed filenames if workflow scripts are run. This is not raw-data overwrite, but it is a reproducibility and artifact-protection risk.

## 12. Project Hazards

- Dirty Git working tree existed before this audit:
  - `.gitignore` modified.
  - Six untracked source/script/test files related to confidence intervals and repeated runs.
- VS Code selected Python interpreter is not detectable from project files.
- Current shell is not activated into `.venv`.
- `requirements.txt` exists but dependencies are unpinned.
- Generated outputs are tracked in Git, including 40 files under `outputs/`.
- Large generated artifact exists in the working tree:
  - `outputs/tables/cleaned_data.csv`, about 118 MB, ignored by `.gitignore`.
- Scripts write to fixed generated-output paths under `outputs/` and can overwrite prior artifacts without versioning.
- `.gitignore` has a duplicate `outputs/tables/cleaned_data.csv` entry.
- `src/` and `tests/` contain `__pycache__` files. These are ignored/generated cache artifacts, not tracked in the inspected status.
- Multiple executable workflow scripts exist, and README does not identify a single primary entry point.
- Raw data is present inside the project folder under `data/raw/`, although ignored and untracked. It should remain protected and should not be copied, reformatted, overwritten, or staged.
- No missing README was found.
- No missing dependency specification was found, though dependency versions are not pinned.
- No missing `.gitignore` was found.
- No committed raw-data files were found under `data/raw/`.
- No obvious hard-coded absolute paths were found in targeted project text scans outside environment files.
- No obvious secrets, tokens, passwords, or credentials were found in targeted text scans. No secret values were displayed.
- No model artifact files were found.
- No GUI code was found.

## 13. Phase 0 Pass/Fail Decision

Decision: PASS WITH BLOCKERS.

Rationale:

- Correct project folder confirmed.
- User-confirmed backup recorded.
- `.venv` exists and a project virtual-environment Python executable was identified.
- Current shell Python was identified.
- Git repository was inspected with read-only commands.
- Raw-data locations were identified.
- Raw-data files were not copied, opened in an editor, imported into the project, rewritten, deleted, moved, or renamed.
- No evidence shows that this audit modified raw data.
- No critical unresolved safety issue prevents further project inspection.
- Blockers remain before Phase 1 because the working tree is dirty, VS Code interpreter selection is not detectable, the current shell is not activated into `.venv`, and generated outputs are tracked/overwritten by fixed script paths.

## 14. Blocking Issues Before Phase 1

- Resolve or explicitly accept the pre-existing dirty Git state before making any Phase 1 changes.
- Confirm that VS Code is using the intended `.venv` interpreter, or explicitly document the accepted interpreter.
- Decide how to protect generated outputs before running workflows on new data, because scripts overwrite fixed `outputs/` paths.
- Keep the external 12-hour Excel datasets outside the project until a controlled import plan is approved.
- Do not run the full pipeline or model training until raw-data and output-protection decisions are made.

## 15. Recommended Next Action

Review this audit file and the current Git status. Before Phase 1, confirm the intended Python interpreter and decide how to handle the pre-existing dirty Git state and overwrite-prone generated outputs.

## Verification Notes

- Initial `git status --short` was captured before creating this file.
- Raw file size and modification-time metadata were captured during the audit.
- No test suite was run.
- No analysis pipeline was run.
- No model training was run.
- No package management command was run.
- The only file intentionally created by this audit is `docs/phase_0_safety_audit.md`.
