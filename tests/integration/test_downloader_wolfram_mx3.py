from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pytest
from nbclient import NotebookClient

# Run explicitly:
#   RUN_CDAWS_INTEGRATION=1 pytest -m integration tests/integration/test_downloader_wolfram_mx3.py -q
RUN_INTEGRATION = os.getenv("RUN_CDAWS_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not RUN_INTEGRATION, reason="Set RUN_CDAWS_INTEGRATION=1 to run network integration tests."),
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DOWNLOADER_NOTEBOOK = REPO_ROOT / "CDASWS Downloader.ipynb"
SYMPY_NOTEBOOK = REPO_ROOT / "SymPy Kinematic Models.ipynb"
PLOTS_NOTEBOOK = REPO_ROOT / "Plots.ipynb"
SHOCKS_DIR = REPO_ROOT / "Shocks"
PYTHON_NORMALS_PATH = SHOCKS_DIR / "Kinematic Shock Normals.json"
WOLFRAM_NORMALS_PATH = SHOCKS_DIR / "Kinematic Shock Normals (Wolfram).json"

EVENT_SPECS: dict[str, dict[str, dict[str, Any]]] = {
    "2022-01-16 19-09-00": {
        "ace": {"t0": "2022-01-16T18:36:30.437000+00:00", "dt0_u": -8, "dt1_u": -4, "dt0_d": 2, "dt1_d": 5},
        "wind": {"t0": "2022-01-16T18:30:46.500000+00:00", "dt0_u": -8, "dt1_u": -3, "dt0_d": 2, "dt1_d": 7},
        "dscovr": {"t0": "2022-01-16T18:36:03+00:00", "dt0_u": -9, "dt1_u": -2, "dt0_d": 2, "dt1_d": 11},
        "mms1": {"t0": "2022-01-16T19:06:59.176857+00:00", "dt0_u": -10, "dt1_u": -2, "dt0_d": 2, "dt1_d": 12},
    },
    "2023-02-26 19-23-00": {
        "ace": {"t0": "2023-02-26T18:42:28+00:00", "dt0_u": -5, "dt1_u": -2, "dt0_d": 2, "dt1_d": 5},
        "wind": {"t0": "2023-02-26T18:50:40.500000+00:00", "dt0_u": -7, "dt1_u": -2, "dt0_d": 2, "dt1_d": 7},
        "dscovr": {"t0": "2023-02-26T18:43:27+00:00", "dt0_u": -5, "dt1_u": -2, "dt0_d": 2, "dt1_d": 5},
        "mms1": {"t0": "2023-02-26T19:20:14.475565+00:00", "dt0_u": -10, "dt1_u": -2, "dt0_d": 2, "dt1_d": 12},
    },
}

MX3_BASELINE: dict[str, dict[str, list[float]]] = {
    "2022-01-16 19-09-00": {
        "ace": [-0.85094600824723, 0.040656838422794, -0.523677298092602],
        "wind": [-0.499712602041637, -0.259457538606917, -0.826419446177797],
        "dscovr": [-0.918658852577209, -0.220099538564682, -0.328058391809464],
    },
    "2023-02-26 19-23-00": {
        "ace": [-0.897416047907687, -0.228643192072503, -0.377315156966486],
        "wind": [-0.923856666625301, -0.37504943665003, -0.07633334527245],
        "dscovr": [-0.935042858123779, -0.318080544471741, 0.156587034463882],
    },
}


def _unsigned_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))


def _event_to_iso_z(event_key: str) -> str:
    date_part, time_part = event_key.split(" ")
    return f"{date_part}T{time_part.replace('-', ':')}Z"


def _patch_tshock(source_nb: Path, out_nb: Path, tshock_iso_z: str) -> None:
    nb = nbformat.read(source_nb, as_version=4)
    replacement = f'tShock = pd.to_datetime("{tshock_iso_z}")\n'

    patched = False
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        lines = cell.source.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.strip().startswith("tShock = pd.to_datetime("):
                lines[i] = replacement
                cell.source = "".join(lines)
                patched = True
                break
        if patched:
            break

    if not patched:
        raise AssertionError("Could not find tShock assignment in downloader notebook")

    nbformat.write(nb, out_nb)


def _run_notebook(path: Path, workdir: Path, timeout: int = 3600) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(workdir)}},
    )
    client.execute()


def _append_code_cell(path: Path, code: str) -> None:
    nb = nbformat.read(path, as_version=4)
    nb.cells.append(nbformat.v4.new_code_cell(code))
    nbformat.write(nb, path)


def _copy_repo_inputs(root: Path) -> None:
    shutil.copy2(DOWNLOADER_NOTEBOOK, root / DOWNLOADER_NOTEBOOK.name)
    shutil.copy2(SYMPY_NOTEBOOK, root / SYMPY_NOTEBOOK.name)
    shutil.copy2(PLOTS_NOTEBOOK, root / PLOTS_NOTEBOOK.name)
    shutil.copytree(SHOCKS_DIR, root / "Shocks")


def _assert_event_parquets(data_root: Path, event: str) -> None:
    event_dir = data_root / event
    assert event_dir.exists(), f"Missing event output folder: {event_dir}"

    for sat in ["ACE", "Wind", "DSCOVR", "MMS1"]:
        sat_file = event_dir / f"{sat}.parquet"
        assert sat_file.exists(), f"Missing parquet for {sat} in {event}"


@pytest.fixture(scope="session")
def generated_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("downloader-notebook-integration")
    _copy_repo_inputs(root)
    assert (root / "Shocks" / PYTHON_NORMALS_PATH.name).exists()
    assert (root / "Shocks" / WOLFRAM_NORMALS_PATH.name).exists()

    downloader_nb = root / DOWNLOADER_NOTEBOOK.name
    for event in EVENT_SPECS:
        _patch_tshock(downloader_nb, downloader_nb, _event_to_iso_z(event))
        _run_notebook(downloader_nb, workdir=root, timeout=7200)
        _assert_event_parquets(root / "Data", event)

    sympy_nb = root / SYMPY_NOTEBOOK.name
    _append_code_cell(
        sympy_nb,
        """
out = {
    "planar": planar_validation_df.to_dict("records"),
    "spherical": spherical_validation_df.to_dict("records"),
    "spherical_sat": spherical_sat_validation_df.to_dict("records"),
    "summary": summary_df.to_dict("records"),
}
Path("sympy_validation_results.json").write_text(json.dumps(out, indent=2, default=str, sort_keys=True))
""",
    )
    _run_notebook(sympy_nb, workdir=root, timeout=3600)

    plots_nb = root / PLOTS_NOTEBOOK.name
    _append_code_cell(
        plots_nb,
        """
EVENT_SPECS = __EVENT_SPECS__
export_mx3_normals_for_events(EVENT_SPECS, output_path=Path("mx3_results.json"), require_all=False)
""".replace("__EVENT_SPECS__", json.dumps(EVENT_SPECS, indent=2, sort_keys=True)),
    )
    _run_notebook(plots_nb, workdir=root, timeout=3600)

    return root


def test_sympy_notebook_matches_wolfram(generated_workspace: Path) -> None:
    results_path = generated_workspace / "sympy_validation_results.json"
    assert results_path.exists(), "SymPy validation output was not produced"
    assert (generated_workspace / "Shocks" / PYTHON_NORMALS_PATH.name).exists(), "Python kinematic normals export was not produced"
    assert (generated_workspace / "Shocks" / WOLFRAM_NORMALS_PATH.name).exists(), "Wolfram kinematic normals baseline is missing"

    results = json.loads(results_path.read_text())
    planar_rows = results["planar"]
    spherical_rows = results["spherical"]

    planar_events = {row["event"] for row in planar_rows}
    spherical_events = {row["event"] for row in spherical_rows}

    expected_events = set(EVENT_SPECS.keys())
    assert planar_events == expected_events
    assert spherical_events == expected_events

    for row in planar_rows:
        assert row["within_numerical_reason"], f"Planar mismatch for {row['event']}"

    for row in spherical_rows:
        assert row["within_numerical_reason"], f"Spherical mismatch for {row['event']}"


def test_plots_notebook_mx3_matches_baseline(generated_workspace: Path) -> None:
    results_path = generated_workspace / "mx3_results.json"
    assert results_path.exists(), "Plots MX3 output was not produced"

    mx3_results = json.loads(results_path.read_text())

    for event, sats in MX3_BASELINE.items():
        assert event in mx3_results, f"Missing event in MX3 output: {event}"
        for sat, expected in sats.items():
            actual = mx3_results[event].get(sat)
            assert actual is not None, f"MX3 unavailable for {event}/{sat}"
            actual_arr = np.asarray(actual, dtype=float)
            expected_arr = np.asarray(expected, dtype=float)
            assert _unsigned_l2(actual_arr, expected_arr) < 1e-6, f"MX3 mismatch for {event}/{sat}"
