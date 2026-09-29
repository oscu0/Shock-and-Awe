# CDAWeb ACE and STEREO-A archive

Run in the `icme3.12-metal` Conda environment. The example covers 16 complete
calendar years, 2010 through 2025, with an exclusive UTC end date:

```bash
conda run -n icme3.12-metal python cdaweb_archive.py download --start 2010-01-01 --end 2026-01-01
conda run -n icme3.12-metal python cdaweb_archive.py merge --start 2010-01-01 --end 2026-01-01
```

`download` writes independent Parquet chunks under `Data/CDAWeb Archive/chunks/`.
Both missions use calendar-month chunks. The hourly STEREO product keeps each
request small.
Each chunk has a JSON manifest with its interval, dataset IDs, row and valid
value counts, and SHA-256. Rerunning the same command skips chunks whose
manifest and checksum match. Failed chunks are reported and left incomplete;
rerun `download` to retry them. The command exits nonzero while any requested
chunk remains incomplete. CDAWeb's explicit HTTP 404 no-data response creates
an empty, marked chunk rather than inventing measurements. `merge` runs
separately, refuses missing or damaged
chunks, and streams each mission into one Parquet in `merged/` without loading
the full 16 years into memory. Use the same dates, mission, and output directory
for both commands. `--mission ace` or `--mission stereo-a` limits either run.

STEREO-A comes from the single hourly `STA_COHO1HR_MERGED_MAG_PLASMA` product:
`BR`, `BT`, `BN`, `B`, `plasmaSpeed`, `plasmaDensity`, `plasmaTemp`, and its HGI
position fields. The RTN field and HGI spacecraft position are converted to
GSE at each sample. This archive cadence is for long-period solar-wind work;
the notebook's higher-cadence MAG path remains available for shock analysis.
The COHO position fields are rounded (for example, radius to 0.01 AU), so
these GSE positions and the orientation used for rotation have the product's
coordinate precision. Duplicate source epochs are reduced to the last record
in CDAWeb order, with the number dropped recorded in each manifest.
ACE combines hourly `AC_H2_MFI` with `AC_H2_SWE` before 2018-01-01 and
`AC_K1_SWE` from that date onward, following the notebook's archive handoff.
`AC_K1_SWE` is labeled preliminary/browse by CDAWeb and has no GSE velocity
vector; those output components remain null. The switch is a calendar/model
coefficient boundary, not an exact Carrington-rotation boundary.

Stored magnetic fields are in nT, speed and velocity in km/s, density in
cm⁻³, temperature in K, and GSE spacecraft positions and radial distance in
Earth radii (6378 km). Scalar `B` is the source's average of field magnitudes;
it need not equal the norm of the averaged vector. Both products can have
missing values. The chunk manifest's valid counts make that visible without
silently filling gaps.

The script defaults to four attempts per chunk and a 180-second CDAWeb request
timeout. `--attempts`, `--retry-wait`, and `--request-timeout` can be adjusted
for service conditions. These settings do not change the chunk identity.
