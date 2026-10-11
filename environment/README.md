# Environment notes

## Recorded training environment

The article records Python 3.10.9, PyTorch 2.1.0.dev20230521, pandas 1.5.3,
and NumPy 1.21.5 for its results environment. Saved September 14 fitting-plan,
replay-gate, and completed-run records confirm these versions for all 90
additional revision fits, using CPU execution and one PyTorch thread.

- `revision-runtime-evidence.json` records the common runtime, source hashes,
  and the scope of the historical replay checks.
- `revision-run-runtime-inventory.csv` maps the 90 fits to their saved summary
  hashes and recorded runtime fields.
- `declared-results-environment.yml` records the article's declared versions
  and the other dependencies required by the source.

The JSON/CSV runtime projections omit machine-specific interpreter paths and
identify their transformations. They derive from saved experiment records,
not a present-day package query, and do not replace canonical fitting records.

Exact historical SciPy, Matplotlib, PyArrow, and PyYAML versions were not
recorded in those fit summaries. Their historical pins remain unavailable.
The PyTorch development wheel/container is not bundled or claimed to remain
available from a package index. The declared environment is therefore **not a
complete historical lock**, and no bit-for-bit reproduction guarantee is made.

## Source compatibility and reporting checks

`requirements-ci.txt` and `pyproject.toml` at repository root define
source-compatibility requirements, not historical dependency pins.
`report-smoke-environment.json` records the separately dated environment used
for aggregate-report regeneration and software checks; it must not be treated
as training provenance.

Two fixed-data replay checks matched their historical scalar and per-user
outputs in the original runtime. These bounded checks do not establish exact
reproduction of every fit across environments or hardware. See
[`docs/REVISION_REPRODUCTION.md`](../docs/REVISION_REPRODUCTION.md) for the
supported reporting workflow and required external artifacts.
