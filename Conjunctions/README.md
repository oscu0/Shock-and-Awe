# Conjunction–CME Catalog Matching

Run `../MESSENGER Conjunctions.ipynb` with the `icme3.12-metal` kernel. Its
`ROOT_FOLDER` variable points to the attached **Mercury storm** Google Drive
folder. The notebook reads copied inputs from `Inputs` and writes new tables
to `Conjunction Analysis`. It can run independently of the current directory.

## Terms

| Term | Meaning |
| --- | --- |
| Conjunction Day | One daily geometry sample from the input table. |
| Conjunction | A continuous run of consecutive input dates, including overlapping or changing partners. |
| Partner | A planet or spacecraft listed in the final input column. |
| CME | A catalog entry, including a shock-only CME. |
| Match | One conjunction linked to one catalog CME. |

Conjunction starts and ends are the first and last supplied daily timestamps,
with no extra day appended. Missing days break a conjunction. Overlapping or
changing partners retain one shared ID. Partner participation intervals are
preserved inside that period so each partner's matching uses its actual dates.
The previous CONJ-003 and CONJ-004 are now one CONJ-003, covering 2011-05-17
through 2011-06-08, with STEREO-A and Venus participation.

## Inputs

`Inputs/messenger_conjunctions.csv` is the daily geometry table. The original
workbook is copied as `Inputs/mercury_conj.xlsx`; its older Earth-CME selection
does not control this analysis. `Inputs/merged_icme_short.csv` is copied from
the requested SINP export in ICME_list.

`Inputs/HELIO4CAST/HELIO4CAST_ICMECAT_v23.csv` is the complete publisher source.
The original parameter-header filename is also preserved. Per-spacecraft
subsets use names such as `HELIO4CAST MESSENGER 2011-2015.csv`.
`Inputs/Input Sources.csv` records the copied input paths and SHA-256 hashes.
Existing source files in Mercury storm were not renamed or edited.

`Inputs/OMNI.parquet` contains the complete ICME_list hourly OMNI series:
267,481 samples from 1996-01-01 to 2026-07-07 UTC. It is a format conversion
of `Inputs/omni_v_dst.pkl`, with numeric types and UTC timestamps and without
resampling, interpolation, or row removal. Magnetic components are GSE, in nT;
Dst is in nT. `Inputs/merged_icme_omni.csv` is the current full merged SINP
list. Its short-list predecessor remains the selection source for matching,
so the existing CME IDs and memberships are preserved.
`Inputs/OMNI Input Sources.csv` records source paths, preparation, and hashes.

## Matching

All timestamps are interpreted as UTC. MESSENGER CMEs match the inclusive
interval from `conjunction_start − 24 h` to `conjunction_end + 24 h`. Partner
CMEs use their participation interval(s) with the same tolerance. Each
conjunction/catalog/CME association is retained once.

- HELIO4CAST CMEs use `mo_start_time` as the CME start.
- SINP CMEs with a body start use `T_start`.
- SINP shock-only CMEs use `T_shock`, with `cme_type = Shock-Only CME`.
  Their missing body starts remain missing.
- MESSENGER CMEs and partner CMEs are matched independently to the conjunction.
  There is no Mercury-CME/partner-CME association or propagation calculation.

For Earth, the catalog comparison uses `T_start` against Wind `mo_start_time`
for CMEs with body starts, or `T_shock` against Wind `icme_start_time` for
shock-only CMEs. Corresponding boundaries must differ by at most 24 hours and
both must fall inside the tolerated Earth participation interval.

## Results

| File in Conjunction Analysis | Contents |
| --- | --- |
| Conjunctions.csv | Continuous conjunctions with starts, ends, partners, and day counts. |
| Conjunction Days.csv | Original daily samples assigned to conjunction IDs. |
| Partner Intervals.csv | Each partner's actual participation windows under the shared IDs. |
| Conjunction CME Matches.csv | One row per conjunction–catalog–CME match. |
| Conjunction Summary.csv | Independent MESSENGER and partner CME counts for every conjunction. |
| Catalog Summary.csv | Unique CME counts, matched conjunction counts, and association counts by catalog. |
| Earth Catalog Agreement.csv | Earth catalog comparisons using matching physical boundaries. |
| Conjunction CME Table.csv | The requested 11-column view, one row per CME match or unmatched conjunction. |
| CME Table Sources.csv | Row numbers, conjunction IDs, catalog CME IDs, matching boundaries, and shock times for that view. |
| OMNI Supplements.csv | Missing Earth parameters filled from OMNI, with windows, sample counts, source values, and table-row links. |

A CME can match several conjunctions, and several CMEs can match a conjunction.
Wind and SINP counts are kept separate because they may describe the same
physical CME. The SINP shock-only count is included in the total SINP count.
Rerunning the export cell replaces the generated tables in Conjunction Analysis.

## Requested CME Table

MESSENGER and partner CMEs occupy separate rows; the opposite spacecraft's
CME columns are empty. For a MESSENGER or unmatched row, `partner_spacecraft`
lists the conjunction's participants. Partner rows name the catalog observer;
VEX is displayed as Venus Express. Every conjunction is represented, including
those with no matched CMEs.

`messenger_cme_bmax` and `partner_cme_bmax` use HELIO4CAST `mo_bmax`, the
maximum total magnetic magnitude within the magnetic obstacle, in nT. Body
starts and ends use `mo_start_time` and `mo_end_time`. Missing Earth Bmax
values are supplemented from the hourly OMNI vectors as described below.

`partner_cme_vmean` uses HELIO4CAST `mo_speed_mean` or SINP
`v_average_omni_body`, in km/s. HELIO4CAST means cover the magnetic
obstacle interval; SINP means retain their catalog OMNI window. VEX has
no plasma parameters in HELIO4CAST, so its mean speed stays empty.
SINP has no Bmax field in the source CSV. Dst uses SINP `dst_min_omni_body`,
in nT, with missing Earth-CME values supplemented from hourly OMNI. Earth
rows with unavailable Dst contain `Missing`;
non-Earth rows have an empty Dst cell. CME-averaged components are not used
to estimate Bmax. Wind and SINP retain their source-specific
values on separate rows; possible counterparts are in Earth Catalog Agreement.

Shock-only CMEs retain empty body-start/end cells; their shock timestamps are
available in CME Table Sources. The table-row numbers link that file to the
11-column view without adding unrequested columns.

## OMNI Supplements

Only missing Earth-CME Bmax and Dst are filled. Existing HELIO4CAST and SINP
values are retained. OMNI cannot fill measurements at MESSENGER, Venus Express,
or STEREO. Rows without an Earth CME retain `Missing` Dst; no conjunction-wide
minimum is substituted for a CME statistic.

Follow ICME_list's `process_omni` / `GetParFromOMNI` parameter windows: one hour
before the CME body start through one hour after its end, inclusively. A
shock-only CME uses shock minus one hour through shock plus 21 hours. If a
body start is known but its end is absent, the upper limit is start plus 20
hours, matching the producer's fallback.

Bmax is the maximum of `sqrt(bx**2 + by**2 + bz**2)` over complete hourly
vectors. Dst is the minimum hourly value. Documented OMNI fills (999.9 for B
components and 99999 for Dst) are masked, and incomplete vectors remain missing.
The hourly-vector maximum differs from both scalar magnetic-magnitude averages
and HELIO4CAST's higher-cadence MO maximum. Each supplement records the window,
valid sample counts, derived statistics, previous values, and changed fields.

The current OMNI/full merged SINP snapshots contain revised Dst values compared
with the older short SINP list. Populated table values are preserved; the audit
retains the current OMNI minimum beside the previous SINP value.

The repository's existing `inputs` and `results` directories preserve the
previous analysis as a historical baseline; the revised notebook does not use
or overwrite them.
