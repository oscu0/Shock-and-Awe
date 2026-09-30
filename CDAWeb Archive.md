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
The requested start and end apply to every mission. A request outside a
product's reported coverage prints a warning rather than changing the range.
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
`SOLO_COHO1HR_MERGED_MAG_PLASMA`. Their usable hourly windows begin at
2007-01-01 for STEREO-B, 2018-08-12 09:00 UTC for PSP, and
2020-02-10 06:00 UTC for Solar Orbiter. STEREO-B's last catalog hour is
2014-09-27 16:00 UTC.
Chunks wholly outside these known windows are saved empty without a CDAWeb
request. A chunk crossing a known boundary asks CDAWeb only for the overlapping
hours while retaining the full requested chunk interval. PSP's first hourly
query must begin at 09:00 UTC on launch day because Horizons has no PSP
ephemeris before 08:16 TDB; querying the whole August 2018 month otherwise
fails repeatedly during coordinate conversion.
Merged output uses the requested date range in its filename and contains no
fabricated hourly rows outside the available data. STEREO-B merges can reuse
the earlier shortened final chunk and skip out-of-lifetime months even if
empty chunks have not been downloaded. PSP and Solar Orbiter have no fixed end
in the script; CDAWeb coverage is checked at the start of each download.
The STEREO-B product supplies scalar plasma speed only. PSP and Solar Orbiter
supply RTN plasma velocity components, which are
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
Scalar field magnitude, speed, density, and temperature values at or below
zero are stored as null, including the `AC_K1_SWE` `-999.9` and zero fills.
Signed vector components are retained, including valid zero components.

The script defaults to four attempts per chunk and a 180-second CDAWeb request
timeout. `--attempts`, `--retry-wait`, and `--request-timeout` can be adjusted
for service conditions. These settings do not change the chunk identity.
