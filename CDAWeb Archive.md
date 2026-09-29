# CDAWeb hourly heliospheric archive

Run in the `icme3.12-metal` Conda environment. The example covers 16 complete
calendar years, 2010 through 2025, with an exclusive UTC end date:

```bash
conda run -n icme3.12-metal python cdaweb_archive.py download --start 2010-01-01 --end 2026-01-01
conda run -n icme3.12-metal python cdaweb_archive.py merge --start 2010-01-01 --end 2026-01-01
```

The default `--mission both` covers STEREO-A and ACE, as in the original
16-year run. To add STEREO-B, PSP, and Solar Orbiter after that run, use:

```bash
conda run -n icme3.12-metal python cdaweb_archive.py download --mission all --start 2010-01-01 --end 2026-01-01
conda run -n icme3.12-metal python cdaweb_archive.py merge --mission all --start 2010-01-01 --end 2026-01-01
```

`all` reuses verified STEREO-A and ACE chunks. Each mission can also run alone
with `--mission stereo-a`, `stereo-b`, `ace`, `psp`, or `solo`. `download` writes
independent calendar-month Parquet chunks under `Data/CDAWeb Archive/chunks/`.
Each chunk has a JSON manifest with its interval, dataset IDs, row and valid
value counts, and SHA-256. Rerunning the same command skips chunks whose
manifest and checksum match. Failed chunks are reported and left incomplete;
rerun `download` to retry them. The command exits nonzero while any requested
chunk remains incomplete. CDAWeb's explicit HTTP 404 no-data response creates
an empty, marked chunk rather than inventing measurements. `merge` runs
separately, refuses missing or damaged
chunks, and streams each mission into one Parquet in `merged/` without loading
the full 16 years into memory. Use the same dates, mission, and output directory
for both commands.

STEREO-A comes from the single hourly `STA_COHO1HR_MERGED_MAG_PLASMA` product:
`BR`, `BT`, `BN`, `B`, `plasmaSpeed`, `plasmaDensity`, `plasmaTemp`, and its HGI
position fields. The RTN field and HGI spacecraft position are converted to
GSE at each sample. This archive cadence is for long-period solar-wind work;
the notebook's higher-cadence MAG path remains available for shock analysis.
The COHO position fields are rounded (for example, radius to 0.01 AU), so
these GSE positions and the orientation used for rotation have the product's
coordinate precision. Duplicate source epochs are reduced to the last record
in CDAWeb order, with the number dropped recorded in each manifest.

The additional hourly sources are `STB_COHO1HR_MERGED_MAG_PLASMA`,
`PSP_COHO1HR_MERGED_MAG_PLASMA`, and
`SOLO_COHO1HR_MERGED_MAG_PLASMA`. Their requests are clipped to the mission
windows: STEREO-B from 2007-01-01 through its last catalog hour at
2014-09-27 16:00 UTC, PSP from 2018-08-12, and Solar Orbiter from 2020-02-10.
The latter two have no fixed end in the script; CDAWeb coverage is checked at
the start of each download. The STEREO-B product supplies scalar plasma speed
only. PSP and Solar Orbiter supply RTN plasma velocity components, which are
rotated to GSE when present. Plasma availability can differ sharply from MAG
availability; for example, PSP's April 2021 monthly sample had 211 valid
velocity vectors out of 720 hourly rows.

The new missions use JPL Horizons spacecraft positions at each source epoch.
Their COHO HGI longitude and latitude (degrees) and radial distance (AU) are
checked against those positions after conversion to heliographic Stonyhurst;
the largest separation is recorded in the chunk manifest. This avoids using
the rounded COHO positions for RTN-to-GSE rotation or archived coordinates.
STEREO-A retains its original COHO position method so already downloaded
chunks remain compatible. For all missions, stored `X_GSE`, `Y_GSE`, and
`Z_GSE` are geocentric spacecraft positions, while `radialDistance` is
heliocentric distance. Both use Earth radii.

ACE combines hourly `AC_H2_MFI` with `AC_H2_SWE` before 2018-01-01 and
`AC_K1_SWE` from that date onward, following the notebook's archive handoff.
`AC_K1_SWE` is labeled preliminary/browse by CDAWeb and has no GSE velocity
vector; those output components remain null. The switch is a calendar/model
coefficient boundary, not an exact Carrington-rotation boundary.

Stored magnetic fields are in nT, speed and velocity in km/s, density in
cm⁻³, temperature in K, and GSE spacecraft positions and heliocentric distance
in Earth radii (6378 km). Scalar `B` is the source's average of field magnitudes;
it need not equal the norm of the averaged vector. All products can have
missing values. The chunk manifest's valid counts make that visible without
silently filling gaps.

The script defaults to four attempts per chunk and a 180-second CDAWeb request
timeout. `--attempts`, `--retry-wait`, and `--request-timeout` can be adjusted
for service conditions. These settings do not change the chunk identity.
