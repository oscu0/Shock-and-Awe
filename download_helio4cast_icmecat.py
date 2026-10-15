"""Download per-spacecraft HELIO4CAST ICMECAT v2.3 subsets.

Run with the icme3.12-metal environment, for example:
    conda run -n icme3.12-metal python download_helio4cast_icmecat.py

The catalog authors' heliocats repository mirrors the published catalog at
https://helioforecast.space/icmecat. Selection uses ICME arrival/start time
(UTC), not magnetic-obstacle start time. The default 2011–2015 interval
covers the MESSENGER conjunction table. By default, files are saved under
HELIO4CAST beside this script, independently of the working directory.
All source columns, row order, numeric strings, timestamps, empty fields,
and the unnamed source index column are preserved. No coordinate or unit
conversions are performed.
Existing files are never overwritten.
"""

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


BASE_URL = "https://raw.githubusercontent.com/cmoestl/heliocats/master/icmecat"
CATALOG_NAME = "HELIO4CAST_ICMECAT_v23.csv"
HEADER_NAME = "HELIO4CAST_ICMECAT_v23_header.txt"
SPACECRAFT = {
    "STEREO-A": "STEREO-A",
    "STEREO-B": "STEREO-B",
    "VEX": "Venus_Express",
    "MESSENGER": "MESSENGER",
    "Wind": "Wind",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "HELIO4CAST",
    )
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2015)
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year")
    start_time = datetime(args.start_year, 1, 1, tzinfo=timezone.utc)
    stop_time = datetime(args.end_year + 1, 1, 1, tzinfo=timezone.utc)
    year_label = f"{args.start_year}-{args.end_year}"
    output_dir = args.output_dir.expanduser().resolve()

    # Download both authoritative files before creating any output files.
    with urlopen(f"{BASE_URL}/{CATALOG_NAME}", timeout=60) as response:
        catalog_bytes = response.read()
    with urlopen(f"{BASE_URL}/{HEADER_NAME}", timeout=60) as response:
        header_bytes = response.read()
    downloaded_at = datetime.now(timezone.utc).isoformat()

    table = list(csv.reader(io.StringIO(catalog_bytes.decode("utf-8-sig"))))
    columns, rows = table[0], table[1:]
    spacecraft_column = columns.index("sc_insitu")
    start_column = columns.index("icme_start_time")
    id_column = columns.index("icmecat_id")
    if any(len(row) != len(columns) for row in rows):
        raise ValueError("Catalog has inconsistent row widths")
    if len({row[id_column] for row in rows}) != len(rows):
        raise ValueError("Catalog contains duplicate ICME identifiers")
    starts = [datetime.fromisoformat(row[start_column]) for row in rows]
    if any(start.utcoffset() is None for start in starts):
        raise ValueError("Catalog contains a start time without a timezone")

    files = {
        f"source/{CATALOG_NAME}": catalog_bytes,
        HEADER_NAME: header_bytes,
    }
    summaries = {}
    for spacecraft, filename_label in SPACECRAFT.items():
        selected = [
            row for row, start in zip(rows, starts)
            if row[spacecraft_column] == spacecraft and start_time <= start < stop_time
        ]
        if not selected:
            raise ValueError(f"No events selected for {spacecraft}")
        filename = f"HELIO4CAST_ICMECAT_v23_{filename_label}_{year_label}.csv"
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(selected)
        files[filename] = buffer.getvalue().encode("utf-8")
        summaries[spacecraft] = {
            "file": filename,
            "events": len(selected),
            "events_by_year": {
                str(year): sum(row[start_column].startswith(str(year)) for row in selected)
                for year in range(args.start_year, args.end_year + 1)
            },
            "first_icme_start_time": min(row[start_column] for row in selected),
            "last_icme_start_time": max(row[start_column] for row in selected),
        }

    metadata = {
        "catalog": "HELIO4CAST ICMECAT v2.3",
        "catalog_page": "https://helioforecast.space/icmecat",
        "catalog_url": f"{BASE_URL}/{CATALOG_NAME}",
        "header_url": f"{BASE_URL}/{HEADER_NAME}",
        "downloaded_at_utc": downloaded_at,
        "source_events": len(rows),
        "columns": columns,
        "selection": f"{start_time.isoformat()} <= icme_start_time < {stop_time.isoformat()}",
        "preservation": "All source columns and values, missing fields, and row order retained; no unit/frame conversions.",
        "spacecraft": summaries,
        "sha256": {
            filename: hashlib.sha256(data).hexdigest()
            for filename, data in files.items()
        },
    }
    files["download_metadata.json"] = (json.dumps(metadata, indent=2) + "\n").encode("utf-8")
    for filename in files:
        if (output_dir / filename).exists():
            raise FileExistsError(f"Refusing to overwrite {output_dir / filename}")
    for filename, data in files.items():
        path = output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(data)
        if path.read_bytes() != data:
            raise ValueError(f"Written file differs from prepared data: {path}")

    for spacecraft, summary in summaries.items():
        print(f"{spacecraft}: {summary['events']} events {summary['events_by_year']}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
