# MESSENGER conjunction ICME analysis

Open `../MESSENGER Conjunctions.ipynb` from the Shock-and-Awe repository root
with the `icme3.12-metal` kernel and run the cells in order. The notebook uses
repository-local inputs and writes its tables to `Conjunctions/results`.
Rerunning the export cell replaces those generated result tables; input
snapshots are read-only to the analysis.

`inputs/messenger_conjunctions.csv` and the `messenger_conjunctions` sheet of
`inputs/mercury_conj.xlsx` contain identical daily conjunction geometry from
2011-03-18 to 2015-04-13. The notebook checks that equality. The workbook's
earlier Earth selection is retained and compared with the new results.
`inputs/merged_icme_short.csv` is a byte-for-byte snapshot of the requested
SINP export in the adjacent ICME_list repository. Input origins and SHA-256
hashes are recorded in `inputs/input_provenance.json`.

HELIO4CAST inputs are in `../HELIO4CAST`, including all five spacecraft's
2011–2015 subsets, the complete v2.3 source snapshot, parameter definitions,
and download metadata. The downloader defaults to that directory, resolved
relative to its own script location:

```bash
conda run -n icme3.12-metal python download_helio4cast_icmecat.py
```

It refuses to overwrite existing files. To fetch a new comparison snapshot,
choose a new directory with `--output-dir`; optional `--start-year` and
`--end-year` set the inclusive download years. The download subsets use
`icme_start_time` for year selection. The notebook checks against the full
source snapshot for any MO-start boundary omissions.

## Matching definition

- Interpret supplied daily timestamps and timezone-naive SINP timestamps as UTC.
- Use HELIO4CAST `mo_start_time` and SINP `T_start` as the ICME starts.
  HELIO4CAST `icme_start_time` and SINP `T_shock` are shock/disturbance times;
  neither substitutes for a missing start.
- Both MESSENGER's start and a listed partner's start must be within the
  inclusive ±24-hour interval around the same daily conjunction timestamp.
- Evaluate every semicolon-separated partner in the final conjunction column.
  Venus maps to VEX, Earth to Wind and SINP, STEREO_A/B to STEREO-A/B.
- Preserve every candidate, then deduplicate repeated daily hits for the same
  MESSENGER/target/catalog/partner event. Wind and SINP remain separate catalog
  evidence. Their Earth candidates are additionally cross-referenced using
  a start-time difference of at most 24 hours.

This screen can miss propagation delays longer than 48 hours between MO starts.
It identifies temporal/geometric candidates, not confirmed common solar CMEs.
The results flag reversed start order, multiple candidates in the same catalog,
and reuse of a partner event across MESSENGER events. Implied radial speeds are
diagnostics based on daily planet distances and MO starts, with no speed filter.

## Saved results

`results/candidate_pairs.csv` is the principal result, with catalog IDs, both
starts, offsets from the representative conjunction day, start delay,
chronology, geometry, ambiguity flags, and supporting-day counts.

The remaining tables contain every daily pairing, per-catalog counts, Earth
cross-catalog agreement, unmatched opportunities, MESSENGER event status,
SINP rows lacking `T_start`, and comparison with the older workbook selection.
`results/analysis_metadata.json` records the matching parameters and file hashes.
The notebook validates the full pairing set using a separate scalar enumeration,
including exact ±24-hour boundary checks.
