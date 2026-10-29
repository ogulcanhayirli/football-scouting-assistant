# Data licences

The MIT licence in LICENSE covers the code only. It does not cover data
from the three sources below, or tables derived from them. This file says,
for each source, what its terms allow, what this repository and the page
publish, and whether anything had to change before the repository is made
public. Checked on 29 October 2026 (prompt 21). I am not a lawyer; this is
my reading of the terms, written down so it can be checked.

What is where:

* **In git:** code, docs, the synthetic test sample in `tests/fixtures/`
  (made up players with ids from 900000), the registered forecasts in
  `forecasts/` (derived tables, CSV), the archetype drafts in
  `archetypes/` (written from the public record, no source data) and two
  small CSVs in `judge/` (short spans of our own generated reports). `data/`
  and `reports/` are gitignored and were never committed (checked over the
  whole history).
* **On the page:** the candidates, bargains and players tables (names,
  ages, clubs, positions, minutes, market values and our model outputs),
  `similar.json` (our style vectors as 8 bit integers, and the measured
  greats' vectors), `reports.json` (our written reports) and the StatsBomb
  logo.
* **Private only:** the raw API Football cache, the Transfermarkt snapshot
  files and the StatsBomb event files, in a private S3 bucket.

## API Football (api-football.com)

* **What the terms allow.** The data is provided to build projects such as
  applications, websites and games. It comes from API Football's partners,
  and reselling it, or selling it directly, to third parties is not
  allowed. I could not open the terms page from the build session (HTTP
  403), so this rests on the published excerpts of
  <https://www.api-football.com/terms>; read the full page once while
  signed in.
* **What we do.** Raw responses stay in the private cache and are never
  published. The repository holds only synthetic fixtures. The page and
  `forecasts/` publish derived tables: per player season figures (minutes,
  share of team minutes) next to our own model outputs. Nothing is sold.
* **Verdict: allowed. Nothing has to leave the repository or the page.**

## transfermarkt-datasets by dcaribou

* **What the licence allows.** The repository
  (github.com/dcaribou/transfermarkt-datasets) carries a CC0 1.0 licence:
  public domain dedication, any use, no attribution required. The data is
  built from Transfermarkt (transfermarkt.com), whose own terms are not
  waived by dcaribou's CC0. Using it for a non commercial project, credited,
  is how the dataset is commonly used.
* **What we do.** The snapshot files stay private. The page and
  `forecasts/` publish one market value per player (with its date and
  rank), the position label and, in the bargains table, the contract
  expiry; they are not a copy of the database. Both credit dcaribou and
  Transfermarkt.
* **Changed in this prompt.** The old README said the dataset "states no
  license". It does: CC0 1.0. The README now says so. The page's footer,
  which says the data is used for non commercial purposes only and
  credited, is still true and was left as it is.
* **Verdict: allowed. Nothing has to leave.** The remaining risk is
  Transfermarkt's own rights over its valuations; if Transfermarkt objects,
  the value columns come off the page.

## StatsBomb open data

* **What the agreement allows.** The StatsBomb Public Data User Agreement
  (LICENSE.pdf in github.com/statsbomb/open-data, last updated 8 September
  2023) gives access for analysis and research. Analysis and conclusions
  may be shared publicly. The user may not distribute, reproduce, sell or
  provide the data to any third party, and may not commercially exploit the
  data or any analysis derived from it. Any publication of analysis must
  carry the StatsBomb logo. StatsBomb asks every user to register at
  statsbomb.com/resource-centre.
* **What we do.** The event files stay in the private cache and bucket.
  The repository holds no StatsBomb data: the tests build their own made up
  events in code, and the logo and the agreement are downloaded, not
  committed. The page publishes results computed from the data: for each
  great and age, the season, team, minutes, an ability percentile within
  his peers and a 48 number style vector, with the StatsBomb logo and
  credit in the tab and the footer. MODELS.md quotes aggregate checks
  (correlations, hit rates).
* **Verdict: allowed, as published analysis with the logo. Nothing has to
  leave.** Still open: register at statsbomb.com/resource-centre, which the
  agreement asks for (CLAUDE.md, Open questions).

## Everything else

No site is scraped by this project. No club logos, badges or player photos
are stored or shown: the API Football adapter drops the `logo` and `photo`
URLs. The written reports and the chat's answers are generated with the
Claude API from our own tables.

## People in the data

The tables describe professional footballers from public match records:
names, ages, clubs, minutes, matches missed injured, market values. They
hold no contact details or private data. A player who wants a row removed
can ask through an issue.
