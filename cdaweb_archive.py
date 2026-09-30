"""Resume bounded heliospheric CDAWeb archives, then merge verified chunks.

Examples (run in the icme3.12-metal environment):
    python cdaweb_archive.py download --start 2010-01-01 --end 2026-01-01
    python cdaweb_archive.py merge --start 2010-01-01 --end 2026-01-01

Intervals are half-open UTC. STEREO-A uses one hourly COHO MAG/plasma product.
ACE uses hourly MFI plus one SWE product per chunk. The SWE switch at
2018-01-01 follows the downloader notebook's archive setting.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.coordinates import CartesianRepresentation, SkyCoord
from cdasws import CdasWs
from sunpy.coordinates import frames, get_horizons_coord


EARTH_RADIUS_KM = 6378.0
ACE_SWE_SWITCH = pd.Timestamp("2018-01-01T00:00:00Z")
SCHEMA_VERSION = 2
SCALAR_CLAMP_VERSION = 1
STEREO_VARIABLES = [
    "BR", "BT", "BN", "B", "plasmaSpeed", "plasmaDensity", "plasmaTemp",
    "radialDistance", "heliographicLatitude", "heliographicLongitude",
]
ACE_MFI_VARIABLES = ["Magnitude", "BGSEc", "SC_pos_GSE"]
ACE_H2_VARIABLES = ["Vp", "Np", "V_GSE"]
ACE_K1_VARIABLES = ["Vp", "Np"]
COHO_DATASETS = {
    "stereo-b": "STB_COHO1HR_MERGED_MAG_PLASMA",
    "psp": "PSP_COHO1HR_MERGED_MAG_PLASMA",
    "solo": "SOLO_COHO1HR_MERGED_MAG_PLASMA",
}
HORIZONS_BODIES = {"stereo-b": "STEREO-B", "psp": "Parker Solar Probe", "solo": "Solar Orbiter"}
COHO_POSITION_VARIABLES = ["radialDistance", "heliographicLatitude", "heliographicLongitude"]
COHO_FIELD_VARIABLES = ["BR", "BT", "BN", "B"]
COHO_PLASMA_VARIABLES = {
    "stereo-b": ["plasmaSpeed", "plasmaDensity", "plasmaTemp"],
    "psp": ["VR", "VT", "VN", "ProtonSpeed", "protonDensity", "protonTemp"],
    "solo": ["VR", "VT", "VN", "ProtonSpeed", "protonDensity", "protonTemp"],
}
# Catalog end for the lost spacecraft is fixed; the other mission ends are checked live.
MISSION_WINDOWS = {
    "stereo-b": (pd.Timestamp("2007-01-01T00:00:00Z"), pd.Timestamp("2014-09-27T17:00:00Z")),
    "psp": (pd.Timestamp("2018-08-12T00:00:00Z"), None),
    "solo": (pd.Timestamp("2020-02-10T00:00:00Z"), None),
}


def utc(value):
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def chunk_bounds(start, end, mission):
    cuts = [start]
    cuts.extend(t for t in pd.date_range(start.floor("D"), end, freq="MS", tz="UTC") if start < t < end)
    if mission == "ace" and start < ACE_SWE_SWITCH < end:
        cuts.append(ACE_SWE_SWITCH)
    cuts.append(end)
    cuts = sorted(set(cuts))
    return list(zip(cuts[:-1], cuts[1:]))


def mission_interval(mission, start, end):
    available_start, available_end = MISSION_WINDOWS.get(mission, (None, None))
    clipped_start = max(start, available_start) if available_start is not None else start
    clipped_end = min(end, available_end) if available_end is not None else end
    if clipped_start >= clipped_end:
        return None
    if clipped_start != start or clipped_end != end:
        print(f"CLIP {mission}: {clipped_start}..{clipped_end}", flush=True)
    return clipped_start, clipped_end


def chunk_stem(start, end):
    return f"{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}"


def chunk_paths(root, mission, start, end):
    base = root / "chunks" / mission / chunk_stem(start, end)
    return base.with_suffix(".parquet"), base.with_suffix(".json")


def validate_coverage(missions, start, end, request_timeout):
    intervals = []
    if "stereo-a" in missions:
        intervals.append(("STA_COHO1HR_MERGED_MAG_PLASMA", start, end, pd.Timedelta("1h")))
    if "ace" in missions:
        intervals.append(("AC_H2_MFI", start, end, pd.Timedelta("1h")))
        if start < ACE_SWE_SWITCH:
            intervals.append(("AC_H2_SWE", start, min(end, ACE_SWE_SWITCH), pd.Timedelta("1h")))
        if end > ACE_SWE_SWITCH:
            intervals.append(("AC_K1_SWE", max(start, ACE_SWE_SWITCH), end, pd.Timedelta("1h")))
    for mission, dataset in COHO_DATASETS.items():
        if mission in missions:
            interval = mission_interval(mission, start, end)
            if interval is not None:
                intervals.append((dataset, *interval, pd.Timedelta("1h")))
    cdas = CdasWs(timeout=request_timeout)
    try:
        for dataset, interval_start, interval_end, cadence in intervals:
            matches = cdas.get_datasets(id=dataset)
            if len(matches) != 1:
                raise RuntimeError(f"Could not resolve coverage for {dataset}")
            coverage = matches[0]["TimeInterval"]
            available_start = utc(coverage["Start"])
            available_end = utc(coverage["End"]) + cadence
            if interval_start < available_start or interval_end > available_end:
                raise ValueError(
                    f"{dataset} covers {available_start}..{available_end}; "
                    f"requested {interval_start}..{interval_end}"
                )
    finally:
        cdas.close()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def completed_chunk(root, mission, start, end):
    parquet, manifest = chunk_paths(root, mission, start, end)
    if not parquet.exists() or not manifest.exists():
        return False
    try:
        info = json.loads(manifest.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        info.get("schema_version") == SCHEMA_VERSION
        and info.get("scalar_clamp_version") == SCALAR_CLAMP_VERSION
        and info.get("mission") == mission
        and info.get("start") == start.isoformat()
        and info.get("end") == end.isoformat()
        and info.get("sha256") == sha256(parquet)
    )


def values(data, name, width=None):
    arr = np.asarray(data[name][...], dtype=float)
    if width is not None and (arr.ndim != 2 or arr.shape[1] != width):
        raise ValueError(f"{name} has shape {arr.shape}, expected (N, {width})")
    arr = arr.copy()
    arr[(~np.isfinite(arr)) | (np.abs(arr) >= 1e29) | np.isclose(arr, -99999.0)] = np.nan
    return arr


def positive_values(data, name):
    """Mask fill codes and nonphysical zero/negative scalar measurements."""
    arr = values(data, name)
    arr[arr <= 0] = np.nan
    return arr


def request(cdas, dataset, variables, start, end):
    status, data = cdas.get_data(dataset, variables, start.to_pydatetime(), end.to_pydatetime())
    if status.get("http", {}).get("status_code") == 404 and data is None:
        print(f"NO DATA {dataset} {start}..{end}", flush=True)
        return None
    if status.get("http", {}).get("status_code") != 200 or data is None:
        raise RuntimeError(f"{dataset} {start}..{end}: {status}")
    missing = set(variables) - set(data)
    if missing or "Epoch" not in data:
        raise ValueError(f"{dataset}: missing returned variables {sorted(missing)} or Epoch")
    return data


def epoch_index(data):
    index = pd.DatetimeIndex(pd.to_datetime(np.asarray(data["Epoch"][...]), utc=True), name="Epoch")
    if not index.is_monotonic_increasing:
        raise ValueError("CDAS Epoch is not monotonic")
    return index


def stereo_frame(data):
    if data is None:
        columns = ["B_X_GSE", "B_Y_GSE", "B_Z_GSE", "B", "N_p", "V", "T_p", "X_GSE", "Y_GSE", "Z_GSE", "radialDistance"]
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="Epoch", tz="UTC")).astype(float)
    source_index = epoch_index(data)
    keep = ~source_index.duplicated(keep="last")
    index = source_index[keep]
    frame = pd.DataFrame(index=index)
    field = np.column_stack([values(data, name)[keep] for name in ("BR", "BT", "BN")])
    radius = values(data, "radialDistance")[keep]  # AU
    latitude = values(data, "heliographicLatitude")[keep]  # HGI degrees
    longitude = values(data, "heliographicLongitude")[keep]  # HGI degrees
    frame.attrs["duplicate_epochs_dropped"] = int((~keep).sum())
    position = np.full((len(index), 3), np.nan)
    valid_position = np.isfinite(radius) & np.isfinite(latitude) & np.isfinite(longitude) & (radius > 0)
    if valid_position.any():
        hgi = SkyCoord(
            longitude[valid_position] * u.deg,
            latitude[valid_position] * u.deg,
            distance=radius[valid_position] * u.au,
            frame=frames.HeliocentricInertial(obstime=index[valid_position]),
        )
        hgs = hgi.transform_to(frames.HeliographicStonyhurst(obstime=index[valid_position]))
        position[valid_position] = hgs.cartesian.xyz.to_value(u.au).T
    distance = np.linalg.norm(position, axis=1)
    valid_position &= np.isfinite(position).all(axis=1) & (distance > 0)
    radial = np.full_like(position, np.nan)
    radial[valid_position] = position[valid_position] / distance[valid_position, None]
    tangent = np.cross(np.array([0.0, 0.0, 1.0]), radial)
    tangent_norm = np.linalg.norm(tangent, axis=1)
    valid_position &= tangent_norm > 0
    tangent[valid_position] /= tangent_norm[valid_position, None]
    normal = np.cross(radial, tangent)
    field_hgs = field[:, :1] * radial + field[:, 1:2] * tangent + field[:, 2:3] * normal

    gse_position = np.full_like(position, np.nan)
    gse_origin = np.full_like(position, np.nan)
    gse_field = np.full_like(field, np.nan)
    if valid_position.any():
        times = index[valid_position]
        hgs = frames.HeliographicStonyhurst(obstime=times)
        gse = frames.GeocentricSolarEcliptic(obstime=times)

        def transform(points):
            coord = SkyCoord(CartesianRepresentation(*(points.T * u.au)), frame=hgs)
            return coord.transform_to(gse).cartesian.xyz.to_value(u.au).T

        origin = transform(np.zeros((len(times), 3)))
        gse_origin[valid_position] = origin
        gse_position[valid_position] = transform(position[valid_position])
        valid_field = valid_position & np.isfinite(field_hgs).all(axis=1)
        if valid_field.any():
            # The HGS-to-GSE transform translates the origin; subtract it for a vector.
            local = SkyCoord(
                CartesianRepresentation(*(field_hgs[valid_field].T * u.au)),
                frame=frames.HeliographicStonyhurst(obstime=index[valid_field]),
            )
            transformed = local.transform_to(frames.GeocentricSolarEcliptic(obstime=index[valid_field]))
            gse_field[valid_field] = transformed.cartesian.xyz.to_value(u.au).T - gse_origin[valid_field]

    frame[["B_X_GSE", "B_Y_GSE", "B_Z_GSE"]] = gse_field
    frame["B"] = positive_values(data, "B")[keep]
    frame["N_p"] = positive_values(data, "plasmaDensity")[keep]
    frame["V"] = positive_values(data, "plasmaSpeed")[keep]
    frame["T_p"] = positive_values(data, "plasmaTemp")[keep]
    frame[["X_GSE", "Y_GSE", "Z_GSE"]] = gse_position * (u.au.to(u.km) / EARTH_RADIUS_KM)
    frame["radialDistance"] = radius * (u.au.to(u.km) / EARTH_RADIUS_KM)
    return frame


def precise_coho_frame(data, mission):
    """Use hourly COHO measurements and precise Horizons positions, storing GSE vectors."""
    columns = [
        "B_X_GSE", "B_Y_GSE", "B_Z_GSE", "B", "N_p", "V", "T_p",
        "X_GSE", "Y_GSE", "Z_GSE", "radialDistance",
    ]
    if mission in {"psp", "solo"}:
        columns += ["V_X_GSE", "V_Y_GSE", "V_Z_GSE"]
    if data is None or len(data["Epoch"]) == 0:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="Epoch", tz="UTC")).astype(float)

    source_index = epoch_index(data)
    keep = ~source_index.duplicated(keep="last")
    index = source_index[keep]
    frame = pd.DataFrame(index=index)
    frame.attrs["duplicate_epochs_dropped"] = int((~keep).sum())

    ephemeris = get_horizons_coord(HORIZONS_BODIES[mission], index)
    position_hgs = ephemeris.cartesian.xyz.to_value(u.au).T
    if not np.isfinite(position_hgs).all():
        raise ValueError(f"{mission}: nonfinite Horizons position")

    # Verify the source fields really have the documented HGI/AU semantics.
    radius = values(data, "radialDistance")[keep]
    latitude = values(data, "heliographicLatitude")[keep]
    longitude = values(data, "heliographicLongitude")[keep]
    has_source_position = np.isfinite(radius) & np.isfinite(latitude) & np.isfinite(longitude) & (radius > 0)
    if has_source_position.any():
        hgi = SkyCoord(
            longitude[has_source_position] * u.deg,
            latitude[has_source_position] * u.deg,
            distance=radius[has_source_position] * u.au,
            frame=frames.HeliocentricInertial(obstime=index[has_source_position]),
        )
        source_hgs = hgi.transform_to(
            frames.HeliographicStonyhurst(obstime=index[has_source_position])
        ).cartesian.xyz.to_value(u.au).T
        differences = np.linalg.norm(source_hgs - position_hgs[has_source_position], axis=1)
        max_difference = float(np.max(differences))
        if max_difference > 0.02:
            raise ValueError(f"{mission}: COHO HGI position differs from Horizons by {max_difference:.4f} AU")
        frame.attrs["max_coho_position_difference_au"] = max_difference
    frame.attrs["position_source"] = "JPL Horizons"

    radial = position_hgs / np.linalg.norm(position_hgs, axis=1)[:, None]
    tangential = np.cross([0.0, 0.0, 1.0], radial)
    tangential /= np.linalg.norm(tangential, axis=1)[:, None]
    normal = np.cross(radial, tangential)
    rtn_basis_hgs = np.stack((radial, tangential, normal), axis=2)
    hgs = frames.HeliographicStonyhurst(obstime=index)
    gse = frames.GeocentricSolarEcliptic(obstime=index)
    origin = SkyCoord(
        CartesianRepresentation(*(np.zeros_like(position_hgs).T * u.au)), frame=hgs
    ).transform_to(gse).cartesian.xyz.to_value(u.au).T

    def rotate(raw):
        result = np.full_like(raw, np.nan)
        valid = np.isfinite(raw).all(axis=1)
        if valid.any():
            vector_hgs = np.einsum("nij,nj->ni", rtn_basis_hgs[valid], raw[valid])
            local = SkyCoord(
                CartesianRepresentation(*(vector_hgs.T * u.au)),
                frame=frames.HeliographicStonyhurst(obstime=index[valid]),
            )
            converted = local.transform_to(
                frames.GeocentricSolarEcliptic(obstime=index[valid])
            ).cartesian.xyz.to_value(u.au).T
            result[valid] = converted - origin[valid]
        return result

    magnetic_rtn = np.column_stack([values(data, name)[keep] for name in ("BR", "BT", "BN")])
    frame[["B_X_GSE", "B_Y_GSE", "B_Z_GSE"]] = rotate(magnetic_rtn)
    frame["B"] = positive_values(data, "B")[keep]
    if mission == "stereo-b":
        frame["V"] = positive_values(data, "plasmaSpeed")[keep]
        frame["N_p"] = positive_values(data, "plasmaDensity")[keep]
        frame["T_p"] = positive_values(data, "plasmaTemp")[keep]
    else:
        velocity_rtn = np.column_stack([values(data, name)[keep] for name in ("VR", "VT", "VN")])
        frame[["V_X_GSE", "V_Y_GSE", "V_Z_GSE"]] = rotate(velocity_rtn)
        frame["V"] = positive_values(data, "ProtonSpeed")[keep]
        frame["N_p"] = positive_values(data, "protonDensity")[keep]
        frame["T_p"] = positive_values(data, "protonTemp")[keep]
    position_gse = ephemeris.transform_to(gse).cartesian.xyz.to_value(u.au).T
    frame[["X_GSE", "Y_GSE", "Z_GSE"]] = position_gse * (u.au.to(u.km) / EARTH_RADIUS_KM)
    frame["radialDistance"] = np.linalg.norm(position_hgs, axis=1) * (u.au.to(u.km) / EARTH_RADIUS_KM)
    return frame


def ace_frame(mfi_data, swe_data, swe_dataset):
    mfi_index = epoch_index(mfi_data) if mfi_data is not None else pd.DatetimeIndex([], name="Epoch", tz="UTC")
    mfi = pd.DataFrame(index=mfi_index)
    if mfi_data is not None:
        mfi[["B_X_GSE", "B_Y_GSE", "B_Z_GSE"]] = values(mfi_data, "BGSEc", 3)
        mfi["B"] = positive_values(mfi_data, "Magnitude")
        mfi[["X_GSE", "Y_GSE", "Z_GSE"]] = values(mfi_data, "SC_pos_GSE", 3) / EARTH_RADIUS_KM
    else:
        for column in ["B_X_GSE", "B_Y_GSE", "B_Z_GSE", "B", "X_GSE", "Y_GSE", "Z_GSE"]:
            mfi[column] = pd.Series(dtype=float)

    swe_index = epoch_index(swe_data) if swe_data is not None else pd.DatetimeIndex([], name="Epoch", tz="UTC")
    swe = pd.DataFrame(index=swe_index)
    if swe_data is not None:
        swe["V"] = positive_values(swe_data, "Vp")
        swe["N_p"] = positive_values(swe_data, "Np")
        if swe_dataset == "AC_H2_SWE":
            swe[["V_X_GSE", "V_Y_GSE", "V_Z_GSE"]] = values(swe_data, "V_GSE", 3)
        else:
            swe[["V_X_GSE", "V_Y_GSE", "V_Z_GSE"]] = np.nan
    else:
        for column in ["V", "N_p", "V_X_GSE", "V_Y_GSE", "V_Z_GSE"]:
            swe[column] = pd.Series(dtype=float)
    swe["plasma_source_dataset"] = pd.Series(swe_dataset, index=swe.index, dtype="string")
    duplicate_count = int(mfi.index.duplicated(keep="last").sum() + swe.index.duplicated(keep="last").sum())
    mfi = mfi.loc[~mfi.index.duplicated(keep="last")]
    swe = swe.loc[~swe.index.duplicated(keep="last")]
    joined = mfi.join(swe, how="outer").sort_index()
    joined.attrs["duplicate_epochs_dropped"] = duplicate_count
    return joined


def download_chunk(cdas, mission, start, end):
    """Fetch and normalize one half-open interval for one spacecraft."""
    if mission == "stereo-a":
        data = request(cdas, "STA_COHO1HR_MERGED_MAG_PLASMA", STEREO_VARIABLES, start, end)
        frame = stereo_frame(data)
        sources = ["STA_COHO1HR_MERGED_MAG_PLASMA"]
    elif mission in COHO_DATASETS:
        dataset = COHO_DATASETS[mission]
        variables = COHO_FIELD_VARIABLES + COHO_PLASMA_VARIABLES[mission] + COHO_POSITION_VARIABLES
        request_interval = mission_interval(mission, start, end) if mission == "stereo-b" else (start, end)
        data = request(cdas, dataset, variables, *request_interval) if request_interval is not None else None
        frame = precise_coho_frame(data, mission)
        sources = [dataset, "JPL Horizons"] if data is not None else [dataset]
    elif mission == "ace":
        if start < ACE_SWE_SWITCH < end:
            raise ValueError("ACE chunk crosses the SWE product switch")
        swe_dataset = "AC_H2_SWE" if end <= ACE_SWE_SWITCH else "AC_K1_SWE"
        swe_variables = ACE_H2_VARIABLES if swe_dataset == "AC_H2_SWE" else ACE_K1_VARIABLES
        mfi = request(cdas, "AC_H2_MFI", ACE_MFI_VARIABLES, start, end)
        swe = request(cdas, swe_dataset, swe_variables, start, end)
        frame = ace_frame(mfi, swe, swe_dataset)
        sources = ["AC_H2_MFI", swe_dataset]
    else:
        raise ValueError(mission)
    frame = frame.loc[(frame.index >= start) & (frame.index < end)]
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate {mission} epochs in {start}..{end}")
    return frame, sources


def save_chunk(root, mission, start, end, frame, sources):
    parquet, manifest = chunk_paths(root, mission, start, end)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    temporary_parquet = parquet.with_suffix(".parquet.tmp")
    temporary_manifest = manifest.with_suffix(".json.tmp")
    frame.to_parquet(temporary_parquet)
    info = {
        "schema_version": SCHEMA_VERSION,
        "scalar_clamp_version": SCALAR_CLAMP_VERSION,
        "mission": mission,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "sources": sources,
        "rows": len(frame),
        "no_data": frame.empty,
        "duplicate_epochs_dropped": frame.attrs.get("duplicate_epochs_dropped", 0),
        "position_source": frame.attrs.get("position_source", sources[0]),
        "max_coho_position_difference_au": frame.attrs.get("max_coho_position_difference_au"),
        "valid_b_vectors": int(frame[["B_X_GSE", "B_Y_GSE", "B_Z_GSE"]].notna().all(axis=1).sum()),
        "valid_velocity_vectors": int(frame[["V_X_GSE", "V_Y_GSE", "V_Z_GSE"]].notna().all(axis=1).sum())
        if all(column in frame for column in ("V_X_GSE", "V_Y_GSE", "V_Z_GSE")) else None,
        "valid_speed": int(frame["V"].notna().sum()),
        "valid_density": int(frame["N_p"].notna().sum()),
        "valid_position": int(frame[["X_GSE", "Y_GSE", "Z_GSE"]].notna().all(axis=1).sum()),
        "sha256": sha256(temporary_parquet),
    }
    temporary_manifest.write_text(json.dumps(info, indent=2) + "\n")
    os.replace(temporary_parquet, parquet)
    os.replace(temporary_manifest, manifest)
    return parquet


def download_archive(root, missions, start, end, attempts, retry_wait, request_timeout):
    failures = []
    for mission in missions:
        interval = mission_interval(mission, start, end)
        if interval is None:
            print(f"SKIP {mission}: no overlap with requested interval", flush=True)
            continue
        bounds = chunk_bounds(*interval, mission)
        for number, (chunk_start, chunk_end) in enumerate(bounds, 1):
            if completed_chunk(root, mission, chunk_start, chunk_end):
                print(f"SKIP {mission} {number}/{len(bounds)} {chunk_start}..{chunk_end}", flush=True)
                continue
            for attempt in range(1, attempts + 1):
                cdas = CdasWs(timeout=request_timeout)
                try:
                    print(f"GET  {mission} {number}/{len(bounds)} {chunk_start}..{chunk_end} attempt {attempt}", flush=True)
                    frame, sources = download_chunk(cdas, mission, chunk_start, chunk_end)
                    path = save_chunk(root, mission, chunk_start, chunk_end, frame, sources)
                    print(f"SAVED {len(frame)} rows {path}", flush=True)
                    break
                except Exception as exc:
                    print(f"FAIL {mission} {chunk_start}..{chunk_end}: {type(exc).__name__}: {exc}", flush=True)
                    if attempt == attempts:
                        failures.append((mission, chunk_start, chunk_end))
                    else:
                        time.sleep(retry_wait * 2 ** (attempt - 1))
                finally:
                    cdas.close()
    if failures:
        print(f"{len(failures)} chunks remain incomplete; rerun download to retry them.", flush=True)
        return 1
    return 0


def merge_archive(root, missions, start, end):
    for mission in missions:
        interval = mission_interval(mission, start, end)
        if interval is None:
            if mission != "stereo-b":
                print(f"SKIP {mission}: no overlap with requested interval", flush=True)
                continue
            mission_start, mission_end = start, start
        else:
            mission_start, mission_end = interval
        bounds = chunk_bounds(mission_start, mission_end, mission)
        missing = [(a, b) for a, b in bounds if not completed_chunk(root, mission, a, b)]
        if missing:
            raise RuntimeError(f"{mission}: {len(missing)} missing or invalid chunks; first is {missing[0]}")
        output_start, output_end = (start, end) if mission == "stereo-b" else (mission_start, mission_end)
        output = root / "merged" / f"{mission}_{chunk_stem(output_start, output_end)}.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        if mission == "stereo-b" and not bounds:
            temporary = output.with_suffix(".parquet.tmp")
            precise_coho_frame(None, mission).to_parquet(temporary)
            os.replace(temporary, output)
            print(f"MERGED {mission}: 0 rows -> {output}", flush=True)
            continue
        temporary = output.with_suffix(".parquet.tmp")
        previous_last = None
        total_rows = 0
        writer = None
        try:
            for a, b in bounds:
                part = pd.read_parquet(chunk_paths(root, mission, a, b)[0])
                if not part.index.is_monotonic_increasing or part.index.has_duplicates:
                    raise ValueError(f"{mission}: unsorted or duplicate epochs in {a}..{b}")
                if len(part) and (part.index[0] < a or part.index[-1] >= b):
                    raise ValueError(f"{mission}: row outside chunk interval {a}..{b}")
                if len(part) and previous_last is not None and part.index[0] <= previous_last:
                    raise ValueError(f"{mission}: duplicate or out-of-order epochs between chunks")
                if len(part):
                    previous_last = part.index[-1]
                table = pa.Table.from_pandas(part, preserve_index=True)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema)
                writer.write_table(table.cast(writer.schema))
                total_rows += len(part)
        finally:
            if writer is not None:
                writer.close()
        os.replace(temporary, output)
        print(f"MERGED {mission}: {total_rows} rows -> {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["download", "merge"])
    parser.add_argument("--start", required=True, help="Inclusive UTC start, e.g. 2010-01-01")
    parser.add_argument("--end", required=True, help="Exclusive UTC end, e.g. 2026-01-01")
    parser.add_argument("--mission", choices=["both", "all", "stereo-a", "stereo-b", "ace", "psp", "solo"], default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("Data/CDAWeb Archive"))
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--retry-wait", type=float, default=2.0, help="Initial seconds before exponential retry")
    parser.add_argument("--request-timeout", type=int, default=180, help="Seconds per CDAWeb request")
    args = parser.parse_args()
    start, end = utc(args.start), utc(args.end)
    if start >= end or args.attempts < 1 or args.retry_wait < 0 or args.request_timeout < 1:
        parser.error("Require start < end, attempts >= 1, retry-wait >= 0, and request-timeout >= 1")
    if args.mission == "both":
        missions = ["stereo-a", "ace"]
    elif args.mission == "all":
        missions = ["stereo-a", "stereo-b", "ace", "psp", "solo"]
    else:
        missions = [args.mission]
    if args.mode == "download":
        validate_coverage(missions, start, end, args.request_timeout)
        raise SystemExit(download_archive(args.output_dir, missions, start, end, args.attempts, args.retry_wait, args.request_timeout))
    merge_archive(args.output_dir, missions, start, end)


if __name__ == "__main__":
    main()
