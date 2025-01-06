# mrio

mrio reads and writes multidimensional GeoTIFF files.

- **GitHub Repository**: [https://github.com/tacofoundation/mrio](https://github.com/tacofoundation/mrio)
- **Documentation**: [https://tacofoundation.github.io/mrio/](https://tacofoundation.github.io/mrio/)

## What is a Multidimensional GeoTIFF (mGeoTIFF)?

A Multidimensional GeoTIFF extends the traditional GeoTIFF format by supporting N-dimensional arrays, similar to formats like NetCDF, HDF5, or Zarr. It maintains the simplicity and compatibility of GeoTIFF, offering fast access and the ability to be opened by any GIS software or library that supports the GeoTIFF format.

## What is a Temporal GeoTTIFF?

The Temporal Geo Tag Image File Format (Temporal GeoTIFF) builds upon the Multidimensional GeoTIFF standard by implementing a stricter convention for defining the temporal dimension. It requires a four-dimensional structure with dimensions ordered as follows: (time, band, x, y). The temporal dimension adheres to the [STAC specification](https://stacspec.org/), which includes a start_datetime and a optional end_datetime. The start_datetime and end_datetime are both strings representing according to [RFC 3339, section 5.6](https://datatracker.ietf.org/doc/html/rfc3339#section-5.6), and utilizes the Gregorian calendar as the reference system for time. For additional details, please refer to the [Specification][SPECIFICATION.md].

## Installation

```python
pip install mrio
```

## How to use it?

### Writing a Multidimensional GeoTIFF

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
    "mrio:pattern": "simulation time band lat lon -> (simulation band time) lat lon",
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

### Writing a Temporal GeoTIFF

```python
from datetime import datetime
import pandas as pd
import tacoreader
import numpy as np
import mrio

# 1. Load a dataset that contains images from the same location
cloudsen12_l1c = tacoreader.load("tacofoundation:cloudsen12-l1c")

# 2. Select a any location
roi_idx = "ROI_00103"
s2_roi_images = cloudsen12_l1c[cloudsen12_l1c["roi_id"] == roi_idx]
s2_roi_images.reset_index(drop=True, inplace=True)

# 3. Load the temporal mini-cube
temporal_stack = np.zeros((len(s2_roi_images), 3, 512, 512), dtype=np.uint16)
for index, row in s2_roi_images.iterrows():
    
    print("Dowloading image %s" % row["tortilla:id"])

    # Convert a pd.Series to a Tortilla DataFrame
    # This will no necessary in tacoreader 0.5.4
    row = tacoreader.TortillaDataFrame(pd.DataFrame(row).T)
    
    # Get the snipet of the Sentinel-2 image
    s2_str = row.read(0).read(0)

    # Load the image
    with mrio.open(s2_str) as src:
        s2_image = src.read([4, 3, 2])

    # Store the image in the temporal stack
    temporal_stack[index] = s2_image
    
# 4. Set the Coordinates
transform = mrio.Affine.from_gdal(*row["stac:geotransform"].iloc[0])
crs = row["stac:crs"].iloc[0]
time_coord = [datetime.fromtimestamp(t).strftime('%Y%m%d') for t in s2_roi_images["stac:time_start"].tolist()]
band_coord = ["B04", "B03", "B02"]

# 5. Set the writing parameters
params = {
    'driver': 'GTiff',
    'dtype': 'uint16',
    'width': temporal_stack.shape[-1],
    'height': temporal_stack.shape[-2],
    'compress': 'zstd',
    'zstd_level': 22,
    'predictor': 2,
    'tiled': True,
    'num_threads': 10,
    'interleave': 'pixel',
    'crs': s2_roi_images["stac:crs"].iloc[0],
    'transform': transform,
    'mrio:pattern': 'time band lat lon -> (band time) lat lon',
    'mrio:coordinates': {
        "time": time_coord,
        "band": band_coord
    },
    'mrio:attributes': {
        'time_start': s2_roi_images["stac:time_start"].tolist(),
        'time_end': s2_roi_images["stac:time_end"].tolist(),
    }
}

# 6. Write the temporal stack
with mrio.open("temporal_stack.tif", mode = "w", **params) as src:
    src.write(temporal_stack)
```

## When to use it?

### Multidimensional GeoTIFF (mGeoTIFF or GeoTIFF)

Ideal for smaller data cubes (below 10 GB) when you need:

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
