# mrio

[![codecov](https://codecov.io/gh/tacofoundation/mrio/graph/badge.svg?token=GDDIMU2WQR)](https://codecov.io/gh/tacofoundation/mrio)

mrio is a library that extends [rasterio](https://github.com/rasterio/rasterio) for reading and writing multidimensional GeoTIFF files.

## What is a Multidimensional GeoTIFF?

A Multidimensional Geo Tag Image File Format (mGeoTIFF) extends the traditional GeoTIFF format by supporting 
N-dimensional arrays, similar to NetCDF, HDF5, or Zarr. It maintains the simplicity, maturity, and compatibility
of GeoTIFF, offering fast access and the ability to be opened by any GIS software or library that supports 
the GeoTIFF format. For additional information, please refer to the [Specification](SPECIFICATION.md).

## What is a Temporal GeoTIFF?

The Temporal GeoTIFF builds upon the mGeoTIFF format by adopting a stricter convention definition. A temporal 
GeoTIFF file **MUST** adhere to the following rules:

1. **Dimensions**: The file must include exactly four dimensions with the following names:
    - `time`: The temporal dimension.
    - `band`: The spectral dimension.
    - `x`: The spatial dimension along the x-axis.
    - `y`: The spatial dimension along the y-axis.

2. **Required Metadata Attributes**: The following metadata attributes are mandatory:
    - `md:id`: A unique identifier for the observation.
    - `md:time_start`: The nominal start time of the observation.
    - `md:time_end`: The nominal end time of the observation (optional).   

For additional information, please refer to the [Specification](SPECIFICATION.md).

## Installation

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

### License

This project is licensed under the CC0 1.0 Universal Public Domain Dedication - see the [LICENSE](LICENSE) file for details.

### Changes

See the [CHANGELOG](CHANGELOG.md) file for details.