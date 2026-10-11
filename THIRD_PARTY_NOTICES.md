# Third-party notices

The Apache License 2.0 and CC BY 4.0 grants cover the original material
identified in [NOTICE](NOTICE). They do not override dataset terms, dependency
licenses, or other third-party rights.

## Source datasets

The current working tree contains source metadata, checksums, reconstruction
code, and derived aggregate evidence. It does not supply raw or processed
interaction tables or per-user records. Dataset access and use remain subject
to the provider's terms:

- **MovieLens 1M:** GroupLens Research; consult the research-use terms in the
  archive README. [Dataset page](https://grouplens.org/datasets/movielens/1m/).
- **CiaoDVD:** the LibRec research dataset collection; consult the source
  README and provider terms. [Dataset page](https://www.librec.net/datasets.html).
- **Yelp Open Dataset:** consult the Dataset License Agreement supplied with
  the dataset and applicable Yelp terms.
  [Dataset page](https://business.yelp.com/data/resources/open-dataset/).
- **Epinions timestamped ratings:** the Michigan State University data-mining
  research archive. The source page and accompanying README used for this
  study did not state an explicit redistribution license. Raw and processed
  records are excluded. Cite the dataset works identified in
  `configs/latent_factor_beyond_rmse/epinions_prospective_datasets.yaml`.
  [Source page](https://www.cse.msu.edu/~tangjili/trust.html).

None of these datasets is relicensed under Apache-2.0 or CC BY 4.0. Historical
metadata references do not grant access or redistribution rights.

## Software dependencies

Dependencies are declared, not vendored or relicensed. Each retains its own
license and notices:

- [setuptools](https://github.com/pypa/setuptools) and [wheel](https://github.com/pypa/wheel)
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/) and [pandas](https://pandas.pydata.org/)
- [Apache Arrow / PyArrow](https://arrow.apache.org/)
- [pypdf](https://pypdf.readthedocs.io/)
- [PyYAML](https://pyyaml.org/)
- [SciPy](https://scipy.org/)
- [PyTorch](https://pytorch.org/)
- [pytest](https://pytest.org/)

Refer to the installed distributions and their upstream repositories for the
license text corresponding to each version, including transitive dependencies.
Publisher templates, bibliography styles, and logos are not included in the
current working tree.

## Names and attribution

Third-party names identify their respective datasets, software, publishers, or
services. No endorsement is implied, and no trademark rights are granted.
