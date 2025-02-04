# mrio

[![codecov](https://codecov.io/gh/tacofoundation/mrio-python/graph/badge.svg?token=GDDIMU2WQR)](https://codecov.io/gh/tacofoundation/mrio-python)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)
<a href="https://github.com/astral-sh/ruff">
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/16768318/409556616-365ffedd-6c8d-45c0-9d50-1b522dcad17f.svg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250204%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250204T125334Z&X-Amz-Expires=300&X-Amz-Signature=3595ebe8188b60c78b8bfdbfd08495a5f30e591802e268211ef28302fc85a0b9&X-Amz-SignedHeaders=host"/>
</a>
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