# mrio

[![codecov](https://codecov.io/gh/tacofoundation/mrio/graph/badge.svg?token=GDDIMU2WQR)](https://codecov.io/gh/tacofoundation/mrio)

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