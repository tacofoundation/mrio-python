# mrio

mrio is a library that extends [rasterio](https://github.com/rasterio/rasterio) for reading and writing multidimensional GeoTIFF files.

- **GitHub Repository**: [https://github.com/tacofoundation/mrio](https://github.com/tacofoundation/mrio)
- **Documentation**: [https://tacofoundation.github.io/mrio/](https://tacofoundation.github.io/mrio/)
- **Specification**: [https://tacofoundation.github.io/mrio/](https://tacofoundation.github.io/mrio/specification)
- **Best Practices**: [https://tacofoundation.github.io/mrio/](https://tacofoundation.github.io/mrio/best-practices)
- **Examples**: [https://tacofoundation.github.io/mrio/](https://tacofoundation.github.io/mrio/best-practices)

## What is a Multidimensional GeoTIFF?

A Multidimensional Geo Tag Image File Format (mGeoTIFF) extends the traditional GeoTIFF format by supporting N-dimensional arrays, similar to NetCDF, HDF5, or Zarr. It maintains the simplicity and compatibility of GeoTIFF, offering fast access and the ability to be opened by any GIS software or library that supports the GeoTIFF format.


## What is a Temporal GeoTIFF?


The **Temporal GeoTIFF (tGeoTIFF)** builds upon the mGeoTIFF format by adopting a stricter convention for defining its dimensions. A tGeoTIFF file **MUST** adhere to the following rules:

1. **Dimensions**: The file must include exactly four dimensions in this specific order:
  (1) `time`, (2) `band`, (3) `x`, and (4) `y`. These dimensions must follow the specified naming convention.

2. **Required Metadata Attributes**: The following metadata attributes are mandatory:
   - `md:time_start`: The nominal start time of the observation.
   - `md:time_end`: The nominal end time of the observation.
   - `md:id`: A unique identifier for the observation.

For additional information, please refer to the [Temporal GeoTIFF Specification](SPECIFICATION.md).


## When to use it?

#### Multidimensional or Temporal GeoTIFF

Ideal for machine learning workflows, especially when each sample (or minicube) should be retrieved in a single operation. It also excels at sharing data with non-specialized users, offering seamless access and compatibility with commonly used geospatial tools.

#### NetCDF, HDF5, or Zarr

Ideal for complex data analysis workflows, these formats provide superior flexibility, supporting nested groups and advanced chunking strategies. They are ideal for storing large datacubes with detailed metadata.

## Proof of Concept

We transformed a large <b style="color: #D32F2F;">2 Terabyte</b> dataset of Zipped Zarr files into a streamlined, **cloud-native** <b style="color: #388E3C;">700 GB</b> dataset. We reduced the file size by 60% while improving the compression strategy. This optimization was achieved using TACO and Temporal GeoTIFFs.

<p style="text-align: center; margin-top: 15px;">
  <a href="https://www.google.com/" target="_blank" style="font-size: 20px; color: #FFFFFF; text-decoration: none; background-color: #673AB7; padding: 10px 20px; border-radius: 5px; transition: background-color 0.3s;">
    🌟 Explore the dataset here!
  </a>
</p>

## Installation

```python
pip install mrio
```

## Examples


### Writing a Multidimensional GeoTIFF

The following example demonstrates how to write a Multidimensional GeoTIFF using 
the `mrio` library. It is as simple as using `rasterio`. 

**Hint:** You can obtain better compression ratios by changing the order of the dimensions to be merged (between parentheses). That means changing `simulation time band lat lon -> (simulation band time) lat lon` to `simulation time band lat lon -> (time band simulation) lat lon`. Check the [BEST PRACTICES](BEST_PRACTICES.md) file for more details.

```python
import mrio
import numpy as np
import xarray as xr
import pandas as pd

# 1. Define a 5-dimensional dataset (toy example)
data = np.random.rand(3, 5, 7, 128, 128)
time = list(pd.date_range("2021-01-01", periods=5).strftime("%Y%m%d"))
bands = ["B02", "B03", "B04", "B08", "B11", "B12", "B8A"]
simulations = ["historical", "rcp45", "rcp85"]

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
    "crs": "EPSG:4326",
    "transform": mrio.transform.from_bounds(-76.2, 4.31, -76.1, 4.32, 128, 128),
    "md:pattern": "simulation time band lat lon -> (simulation band time) lat lon",
    "md:coordinates": {
        "time": time,
        "band": bands,
        "simulation": simulations,
    },
    "md:attributes": {  # OPTIONAL: additional attributes to include in the file
        "hello": "world",
    },
}

# 3. Write the data
with mrio.open("image.tif", mode="w", **params) as src:
    src.write(datacube.values)
```

### Reading a Multidimensional GeoTIFF

The following example demonstrates how to read a Multidimensional GeoTIFF using 
the `mrio` library.

```python
import mrio

# 1. Read the data
with mrio.open("image.tif") as src:
    md_meta = src.md_meta
    data_r = src.read()

# 2. Convert the data back to an xr.DataArray (Optional)
datacube_r = xr.DataArray(
    data=data_r,
    dims=md_meta["md:dimensions"],
    coords=md_meta["md:coordinates"],
    attrs=md_meta["md:attributes"]
)

```

### Writing a Temporal GeoTIFF

The following example demonstrates how to write a Temporal GeoTIFF using
the `mrio` library.

```python
from datetime import datetime
import tacoreader
import numpy as np
import xarray as xr
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
    
    # Get the snipet of the Sentinel-2 image
    s2_str = row.read().read(0)

    # Load the image
    with mrio.open(s2_str) as src:
        s2_image = src.read([4, 3, 2])

    # Store the image in the temporal stack
    temporal_stack[index] = s2_image
    
# 4. Set the Coordinates
time_coord = [datetime.fromtimestamp(t).strftime('%Y%m%d') for t in s2_roi_images["stac:time_start"].tolist()]
band_coord = ["B04", "B03", "B02"]

# 5. Set the writing parameters
params = {
    'driver': 'GTiff',
    'dtype': 'uint16',
    'width': temporal_stack.shape[-1],
    'height': temporal_stack.shape[-2],
    'compress': 'zstd',
    'zstd_level': 9, # 22
    'predictor': 2,
    'blockxsize': temporal_stack.shape[-1],
    'blockysize': temporal_stack.shape[-2],
    'tiled': True,
    #'num_threads': 10,
    'interleave': 'pixel',
    'crs': row["stac:crs"],
    'transform': mrio.Affine.from_gdal(*row["stac:geotransform"]),
    'md:pattern': 'time band lat lon -> (band time) lat lon',
    'md:coordinates': {
        "time": time_coord,
        "band": band_coord
    },
    'md:attributes': {
        'md:time_start': s2_roi_images["stac:time_start"].tolist(),
        'md:time_end': s2_roi_images["stac:time_end"].tolist(),
        'md:id': s2_roi_images["tortilla:id"].tolist(),
    }
}

# 6. Write the temporal stack
with mrio.open("temporal_stack.tif", mode = "w", **params) as src:
    src.write(temporal_stack)
```

### Reading a Temporal GeoTIFF

The following example demonstrates how to read a Temporal GeoTIFF using
the `mrio` library.

```python
import mrio
from datetime import datetime

# Read the Multidimensional GeoTIFF
with mrio.open("temporal_stack.tif") as src:
    data_r = src.read()
    md_meta = src.md_meta

# Create a xr.DataArray (Optional)
temporal_stack = xr.DataArray(
    data=data_r,
    dims=md_meta["md:dimensions"],
    coords=md_meta["md:coordinates"],
    attrs=md_meta["md:attributes"]
)

# Convert the time to a datetime object (Optional)
temporal_stack["time"] = [
    datetime.fromtimestamp(t) for t in md_meta["md:attributes"]["md:time_start"]
]
```

### Validation

The `mrio` library includes two functions to validate if a file is a mGeoTIFF or a tGeoTIFF.

```python
import mrio

# If the file is a mGeoTIFF, it will return True
mrio.is_mgeotiff("image.tif")

# If the file is a tGeoTIFF, it will return True
mrio.is_tgeotiff("temporal_stack.tif")
```


### Best Practices

See the [BEST PRACTICES](BEST_PRACTICES.md) file for details.

### Specification

See the [SPECIFICATION](SPECIFICATION.md) file for details.

### License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

### Changes

See the [CHANGELOG](CHANGELOG.md) file for details.
