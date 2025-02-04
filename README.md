# mrio

[![codecov](https://codecov.io/gh/tacofoundation/mrio-python/graph/badge.svg?token=GDDIMU2WQR)](https://codecov.io/gh/tacofoundation/mrio-python)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)
[![Ruff](https://camo.githubusercontent.com/bb88127790fb054cba2caf3f3be2569c1b97bb45a44b47b52d738f8781a8ede4/68747470733a2f2f696d672e736869656c64732e696f2f656e64706f696e743f75726c3d68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f636861726c6965726d617273682f727566662f6d61696e2f6173736574732f62616467652f76312e6a736f6e)](https://github.com/charliermarsh/ruff)
[![Docs](https://img.shields.io/readthedocs/pavics-weaver)](https://tacofoundation.github.io/mrio/)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)





mrio is a library that extends [rasterio](https://github.com/rasterio/rasterio) for reading and writing multidimensional COG files.

### Examples

Using the `xarray`-like read API:

```python
import mrio

tcog_file = "https://huggingface.co/datasets/tacofoundation/mrio-examples/resolve/main/simple.tif"
with mrio.open(tcog_file, engine="numpy") as src:
    ddd = src[1:2, 0:4, ...]
```

Using the `earthengine`-like read API:

```python
import mrio

tensor = ( 
  mrio.Collection("https://huggingface.co/datasets/tacofoundation/mrio-examples/resolve/main/simple.tif")
      .select(["B01", "B02", "B03"])
      .FilterDate("2021-01-05", "2021-03-10")
      .FilterBounds(-76.1, 4.3, -76.1, 4.3)
      .getInfo()
)
```

### Installation

You can install the `mrio` library using `pip`:

```python
pip install mrio
```

or via `conda`:

```python   
conda install -c conda-forge mrio
```

or from source:

```python
git clone git@github.com:tacofoundation/mrio.git
cd mrio
pip install .
```

### Specification

See the [mrio](https://tacofoundation.github.io/mrio/en/specification/multidimensional-geotiff-specification.html) website for details.

### License

This project is licensed under the CC0 1.0 Universal Public Domain Dedication - see the [LICENSE](LICENSE) file for details.

### Changes

See the [CHANGELOG](CHANGELOG.md) file for details.

### Donate

Did you find this project useful? It wouldn't be possible without the support of the 
[GDAL community](https://gdal.org/en/stable/community/index.html). Consider donating directly to 
GDAL maintainers through GitHub Sponsors or via NumFOCUS to support continued development.

| Donation Links | Organization |
| --- | --- |
| [https://numfocus.org/donate-to-gdal](https://numfocus.org/donate-to-gdal) | NumFOCUS (GDAL's fiscal sponsor) |
| [https://github.com/sponsors/nyalldawson](https://github.com/sponsors/nyalldawson) | Nyall Dawson |
| [https://github.com/sponsors/rouault](https://github.com/sponsors/rouault) | Even Rouault |
| [https://github.com/sponsors/hobu](https://github.com/sponsors/hobu) | Howard Butler |