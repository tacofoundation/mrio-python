# mrio

A Python package for reading and writing multidimensional (n-D) GeoTIFF files.

- **GitHub Repository**: [https://github.com/tacofoundation/mrio](https://github.com/tacofoundation/mrio)
- **Documentation**: [https://tacofoundation.github.io/mrio/](https://tacofoundation.github.io/mrio/)

## What is a Multidimensional GeoTIFF (mGeoTIFF)?

A Multidimensional GeoTIFF extends the traditional GeoTIFF format by supporting N-dimensional arrays, similar to formats like NetCDF, HDF5, or Zarr. It maintains the simplicity and compatibility of GeoTIFF, offering fast access and the ability to be opened by any GIS software or library that supports the GeoTIFF format.

## What is a GeoTTIFF (GeoTTIFF)?

The GeoTemporal Tag Image File Format (GeoTTIFF) inherits on the Multidimensional GeoTIFF, but it
imposes a **strict** way to define the temporal dimension. It always MUST be a 4D structure that 
contains the dimensions on the following order: (time, band, x, y).

## Installation

```python
pip install mrio
```

## How to use it?

```python
import mrio
import numpy as np
import xarray as xr
import pandas as pd
import rasterio as rio

# 1. Define a 5-dimensional dataset (toy example)
data = np.random.rand(3, 5, 7, 128, 128)
time = list(pd.date_range("2021-01-01", periods=5).strftime("%Y%m%d"))
bands = ["B02", "B03", "B04", "B08", "B11", "B12", "B8A"]
affine = rio.transform.from_bounds(-76.2, 4.31, -76.1, 4.32, 128, 128)
simulations = ["historical", "rcp45", "rcp85"]
crs = "EPSG:4326"

datacube = xr.DataArray(
    data=data,
    dims=["simulation", "time", "band", "lat", "lon"],
    coords={
        "simulation": simulations,
        "time": time,
        "band": bands,
    },
)

# 2. Define the parameters
params = {
    "driver": "GTiff",
    "dtype": "float32",
    "width": datacube.shape[-1],
    "height": datacube.shape[-2],
    "interleave": "pixel",
    "crs": crs,
    "transform": affine,
    "mrio:strategy": "simulation time band lat lon -> (simulation band time) lat lon",
    "mrio:coordinates": {
        "time": time,
        "band": bands,
        "simulation": simulations,
    },
    "mrio:attributes": {
        "hello": "world",
    },
}

# 3. Write the data
with mrio.open("image.tif", mode="w", **params) as src:
    src.write(datacube.values)

# 4. Read the data
with mrio.open("image.tif") as src:
    data_r = src.read()
    print(src.attributes())
    print(src.profile)
```

## When to use it?

### Multidimensional GeoTIFF (mGeoTIFF or GeoTIFF)

Ideal for smaller data cubes (1–10 GB) when you need:

- **Portability**: Easy to share and open across platforms.
- **Advanced Compression**: Support for efficient storage.
- **Defined Spatial and Temporal Dimensions**: A standardized way to manage these dimensions.
- **GIS Compatibility**: Seamless integration with GIS software.
- **Simple Chunking**: Focused on spatial dimensions.

### NetCDF, HDF5, or Zarr

Best suited for larger data cubes (over 10 GB) when you need:

- **High Performance**: Optimized for very large datasets.
- **Nested Groups**: Support for hierarchical data organization.
- **Complex Chunking**: Flexible strategies for managing multidimensional data.
- **Diverse Array Shapes**: Ability to handle arrays with varying dimensions.
- **Rich Metadata**: Support for more intricate metadata structures.