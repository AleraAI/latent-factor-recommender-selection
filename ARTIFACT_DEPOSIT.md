# Distribution and archival scope

This repository accompanies the article by Mohamed Elrfaey and Thomas Aaron
Gulliver, *Computers* **2026**, *15*(10), 703, published 11 October 2026.
[DOI: 10.3390/computers15100703](https://doi.org/10.3390/computers15100703)
identifies the article. It is not a software or data archive DOI; no separate
archival deposit is recorded here.

## Current package

The distributed working tree contains source code, configurations, original
selection locks, dataset manifests, aggregate results, generated-table test
fixtures, runtime evidence, and verification tools. Historical CSV/JSON records
and lock files are preserved as provenance. Preprocessing manifests are public
projections: repaired-interaction hash lists are replaced with counts, without
changing other protocol fields. `provenance/public_metadata_redactions.json`
records the original and exported hashes. NPZ bootstrap outputs contain
aggregate resampling statistics rather than per-user observations.

Raw datasets, processed interaction tables, per-user records, prediction
vectors, model checkpoints, complete run archives, manuscript source, and
compiled publication files are not supplied. A path or checksum in a historical
record may refer to one of these omitted artifacts; it is not an inventory of
files available in this package.

`manifests/release_files_sha256.csv` inventories the current package, excluding
itself. The package verifier checks this inventory and the distribution rules:

```bash
python scripts/latent_factor/verify_release.py
```

This file describes the working-tree distribution, not the contents of every
historical Git revision. File checksums establish integrity, not permission to
redistribute referenced third-party data.

## Preparing a release

After intentional code or documentation changes, run the tests and refresh the
package inventory. This updates release checksums, not experimental locks;
do not alter frozen evidence to make a check pass.

```bash
python scripts/latent_factor/verify_release.py --write-manifest
python scripts/latent_factor/build_github_release.py --output /path/to/new-release
python /path/to/new-release/scripts/latent_factor/verify_release.py --strict
```

The destination must be new and outside the source checkout. The exporter
validates the source inventory before copying, includes only approved public
files, and checks the resulting package independently.

## Reproduction boundary

Tests, lock checks, table verification, and report regeneration use included
evidence. Full fitting and user-level statistical inference additionally require
authorized source datasets, exact processed/split tables, saved per-user
sufficient statistics, relevant run records, and compatible runtime artifacts.
These cannot be reconstructed faithfully from rounded aggregate results.

Some provenance files are explicitly transformed projections that omit
identifier lists or machine-specific paths. They retain source hashes and
transformation notes but do not replace canonical records in hash-bound
workflows. See [report reproduction](docs/REVISION_REPRODUCTION.md) and
[environment notes](environment/README.md).

## Any future archive

A separate research archive would need its own version, membership manifest,
access terms, and persistent identifier. Expanded run evidence, data tables,
or historical dependency artifacts may be included only if available and
permitted by their source terms. A future archive must not be described as
available until it exists, and the article DOI must not be reused as its
identifier.
