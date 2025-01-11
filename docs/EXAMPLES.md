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
