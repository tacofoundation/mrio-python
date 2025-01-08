# mrio

mrio is a library for reading and writing multidimensional GeoTIFF files, extending [rasterio](https://github.com/rasterio/rasterio) to support multidimensional arrays.

- **GitHub Repository**: [https://github.com/tacofoundation/mrio](https://github.com/tacofoundation/mrio)
- **Documentation**: [https://tacofoundation.github.io/mrio/](https://tacofoundation.github.io/mrio/)
- **Specification**: [https://tacofoundation.github.io/mrio/](https://tacofoundation.github.io/mrio/specification)
- **Best Practices**: [https://tacofoundation.github.io/mrio/](https://tacofoundation.github.io/mrio/best-practices)
- **Examples**: [https://tacofoundation.github.io/mrio/](https://tacofoundation.github.io/mrio/best-practices)

## What is a Multidimensional GeoTIFF?

A Multidimensional Geo Tag Image File Format (mGeoTIFF) extends the traditional GeoTIFF format by supporting N-dimensional arrays, similar to formats like NetCDF, HDF5, or Zarr. It maintains the simplicity and compatibility of GeoTIFF, offering fast access and the ability to be opened by any GIS software or library that supports the GeoTIFF format.

## What is a Temporal GeoTIFF?

The Temporal GeoTIFF refines the mGeoTIFF format by enforcing a more stringent convention for defining its dimensions. It **MUST** include four dimensions in the following order, with the specified naming convention: `(time, band, x, y)`. Additionally, certain attributes must be included in the file. First, `md:time_start`, which represents the nominal start time of acquisition. Second, `md:time_end`, an optional attribute that indicates the nominal end time of the acquisition or composite period. Lastly, each time step must have a unique identifier (`md:id`). These attributes, `md:time_start`, `md:time_end`, and `md:id`, must be stored in the attribute section of the file (`md:attributes`). For further details, refer to the [Temporal GeoTIFF Specification](SPECIFICATION.md).


## When to use it?

#### Multidimensional or Temporal GeoTIFF

Ideal for machine learning workflows, especially when each sample (or minicube) should be retrieved in a single operation. It also excels at sharing data with non-specialized users, offering seamless access and compatibility with commonly used geospatial tools.

#### NetCDF, HDF5, or Zarr

Ideal for complex data analysis workflows, these formats provide superior flexibility, supporting nested groups and advanced chunking strategies. They are ideal for storing large datacubes with detailed metadata.

## Proof of Concept

We transformed a large <b style="color: #D32F2F;">2 Terabyte</b> dataset of Zipped Zarr files into a streamlined, **cloud-native** <b style="color: #388E3C;">700 GB</b> dataset. This optimization was achieved using TACO and Temporal GeoTIFFs.

<p style="text-align: center; margin-top: 15px;">
  <a href="https://www.google.com/" target="_blank" style="font-size: 20px; color: #FFFFFF; text-decoration: none; background-color: #673AB7; padding: 10px 20px; border-radius: 5px; transition: background-color 0.3s;">
    🌟 Explore the dataset here!
  </a>
</p>

## Installation

```python
pip install mrio
```

## How to use it?

The API is similar to rasterio, but includes three additional parameters for writing: `md:pattern`, `md:coordinates`, and `md:attributes`.

- **`md:pattern`**: A string defining the strategy to reshape the data into a 3D array (band, x, y).
- **`md:coordinates`**: A dictionary specifying the coordinates for each dimension.
- **`md:attributes`**: A dictionary of additional metadata attributes to include in the file.

For reading, the `mrio.open` function works similarly to `rasterio.open` but it return an extra `md_meta` attribute with the 
following keys:

- **`md:dimensions`**: The data's dimensions.
- **`md:coordinates`**: The coordinates for each dimension.
- **`md:attributes`**: Metadata attributes stored in the file.
- **`md:pattern`**: The pattern used to reconstruct the original multidimensional shape.
- **`md:coordinates_len`**: The size of each dimension.

### Writing a reading a Multidimensional GeoTIFF

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

# 4. Read the data
with mrio.open("image.tif") as src:
    md_meta = src.md_meta
    data_r = src.read()

# 5. Convert the data back to an xarray DataArray
datacube_r = xr.DataArray(
    data=data_r,
    dims=md_meta["md:dimensions"],
    coords=md_meta["md:coordinates"],
    attrs=md_meta["md:attributes"]
)

# 6. Assert that the data is the same
assert np.allclose(datacube, datacube_r)
```

### Writing a Temporal GeoTIFF

A reproducible and **real-world example** of a Temporal GeoTIFF: using this format reduces file size by 60% compared to storing different time steps in separate files.

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
    

# 7. Read the data
with mrio.open("temporal_stack.tif") as src:
    data_r = src.read()    
    md_meta = src.md_meta

    # Create the xarray
    temporal_stack = xr.DataArray(
        data=data_r,
        dims=md_meta["md:dimensions"],
        coords=md_meta["md:coordinates"],
        attrs=md_meta["md:attributes"]
    )

    # time coordinate from string to datetime
    temporal_stack["time"] = [
        datetime.fromtimestamp(t) for t in md_meta["md:attributes"]["md:time_start"]
    ]
```

### Best Practices

See the [BEST PRACTICES](BEST_PRACTICES.md) file for details.

### Specification

See the [SPECIFICATION](SPECIFICATION.md) file for details.

### License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

### Changes

See the [CHANGELOG](CHANGELOG.md) file for details.
