# mrio

<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="99" height="20">
    <linearGradient id="b" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <mask id="a">
        <rect width="99" height="20" rx="3" fill="#fff"/>
    </mask>
    <g mask="url(#a)">
        <path fill="#555" d="M0 0h63v20H0z"/>
        <path fill="#dfb317" d="M63 0h36v20H63z"/>
        <path fill="url(#b)" d="M0 0h99v20H0z"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
        <text x="31.5" y="15" fill="#010101" fill-opacity=".3">coverage</text>
        <text x="31.5" y="14">coverage</text>
        <text x="80" y="15" fill="#010101" fill-opacity=".3">68%</text>
        <text x="80" y="14">68%</text>
    </g>
</svg>

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