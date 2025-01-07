# mrio

mrio is a library for reading and writing multidimensional GeoTIFF files, extending [rasterio](https://github.com/rasterio/rasterio) to support multidimensional arrays.

- **GitHub Repository**: [https://github.com/tacofoundation/mrio](https://github.com/tacofoundation/mrio)
- **Documentation**: [https://tacofoundation.github.io/mrio/](https://tacofoundation.github.io/mrio/)
- **Specification**: [https://tacofoundation.github.io/mrio/](https://tacofoundation.github.io/mrio/specification)
- **Best Practices**: [https://tacofoundation.github.io/mrio/](https://tacofoundation.github.io/mrio/best-practices)
- **Examples**: [https://tacofoundation.github.io/mrio/](https://tacofoundation.github.io/mrio/best-practices)

## What is a Multidimensional GeoTIFF?

A Multidimensional Geo Tag Image File Format (mGeoTIFF) extends the traditional GeoTIFF format by supporting N-dimensional arrays, similar to formats like NetCDF, HDF5, or Zarr. It maintains the simplicity and compatibility of GeoTIFF, offering fast access and the ability to be opened by any GIS software or library that supports the GeoTIFF format.

## What is a Temporal GeoTTIFF?

The Temporal GeoTIFF builds upon the mGeoTIFF by implementing a stricter convention for defining the temporal dimension. It requires a four-dimensional structure with dimensions ordered as follows: (time, band, x, y). The temporal dimension adheres to the [STAC specification](https://stacspec.org/), which includes a start_datetime and a optional end_datetime. The start_datetime and end_datetime are both strings representing according to [RFC 3339, section 5.6](https://datatracker.ietf.org/doc/html/rfc3339#section-5.6), and utilizes the Gregorian calendar as the reference system for time. For additional details, please refer to the [Specification][SPECIFICATION.md].

## When to use it?

#### Multidimensional or Temporal GeoTIFF (mGeoTIFF or GeoTIFF)

Perfect for small datacubes (< 1 GB).

#### NetCDF, HDF5, or Zarr

Best for larger or complex datacubes (> 1 GB).

<table style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; text-align: center;">
  <thead>
    <tr>
      <th style="background-color: #2E7D32; color: white; padding: 10px; border: 1px solid #ddd;">Feature</th>
      <th style="background-color: #1976D2; color: white; padding: 10px; border: 1px solid #ddd;">HDF5</th>
      <th style="background-color: #1976D2; color: white; padding: 10px; border: 1px solid #ddd;">Zarr</th>
      <th style="background-color: #1976D2; color: white; padding: 10px; border: 1px solid #ddd;">mGeoTIFF</th>
      <th style="background-color: #512DA8; color: white; padding: 10px; border: 1px solid #ddd;">Explanation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="background-color: #5b615c; padding: 10px; border: 1px solid #ddd;">Portability</td>
      <td style="background-color: #FFF59D; padding: 10px; border: 1px solid #ddd;">2</td>
      <td style="background-color: #EF9A9A; padding: 10px; border: 1px solid #ddd;">3</td>
      <td style="background-color: #A5D6A7; padding: 10px; border: 1px solid #ddd;">1</td>
      <td style="padding: 10px; border: 1px solid #ddd;">mGeoTIFF is widely supported and easily shareable.</td>
    </tr>
    <tr>
      <td style="background-color: #5b615c; padding: 10px; border: 1px solid #ddd;">Compression</td>
      <td style="background-color: #FFF59D; padding: 10px; border: 1px solid #ddd;">2</td>
      <td style="background-color: #FFF59D; padding: 10px; border: 1px solid #ddd;">2</td>
      <td style="background-color: #A5D6A7; padding: 10px; border: 1px solid #ddd;">1</td>
      <td style="padding: 10px; border: 1px solid #ddd;">Advanced compression options thanks to GDAL.</td>
    </tr>
    <tr>
      <td style="background-color: #5b615c; padding: 10px; border: 1px solid #ddd;">Coordinates definition</td>
      <td style="background-color: #EF9A9A; padding: 10px; border: 1px solid #ddd;">3</td>
      <td style="background-color: #EF9A9A; padding: 10px; border: 1px solid #ddd;">3</td>
      <td style="background-color: #A5D6A7; padding: 10px; border: 1px solid #ddd;">1</td>
      <td style="padding: 10px; border: 1px solid #ddd;">Strict spatial and temporal definitions.</td>
    </tr>
    <tr>
      <td style="background-color: #5b615c; padding: 10px; border: 1px solid #ddd;">GIS compatibility</td>
      <td style="background-color: #FFF59D; padding: 10px; border: 1px solid #ddd;">2</td>
      <td style="background-color: #FFF59D; padding: 10px; border: 1px solid #ddd;">2</td>
      <td style="background-color: #A5D6A7; padding: 10px; border: 1px solid #ddd;">1</td>
      <td style="padding: 10px; border: 1px solid #ddd;">Because a mGeoTIFF is still a GeoTIFF.</td>
    </tr>
    <tr>
      <td style="background-color: #5b615c; padding: 10px; border: 1px solid #ddd;">Chunking</td>
      <td style="background-color: #A5D6A7; padding: 10px; border: 1px solid #ddd;">1</td>
      <td style="background-color: #A5D6A7; padding: 10px; border: 1px solid #ddd;">1</td>
      <td style="background-color: #EF9A9A; padding: 10px; border: 1px solid #ddd;">3</td>
      <td style="padding: 10px; border: 1px solid #ddd;">mGeoTIFF only supports spatial chunking.</td>
    </tr>
    <tr>
      <td style="background-color: #5b615c; padding: 10px; border: 1px solid #ddd;">Nested Groups</td>
      <td style="background-color: #A5D6A7; padding: 10px; border: 1px solid #ddd;">1</td>
      <td style="background-color: #A5D6A7; padding: 10px; border: 1px solid #ddd;">1</td>
      <td style="background-color: #EF9A9A; padding: 10px; border: 1px solid #ddd;">3</td>
      <td style="padding: 10px; border: 1px solid #ddd;">mGeoTIFF does not support hierarchical data organization.</td>
    </tr>
    <tr>
      <td style="background-color: #5b615c; padding: 10px; border: 1px solid #ddd;">Diverse Array Shapes Support</td>
      <td style="background-color: #A5D6A7; padding: 10px; border: 1px solid #ddd;">1</td>
      <td style="background-color: #A5D6A7; padding: 10px; border: 1px solid #ddd;">1</td>
      <td style="background-color: #EF9A9A; padding: 10px; border: 1px solid #ddd;">3</td>
      <td style="padding: 10px; border: 1px solid #ddd;">mGeoTIFF only supports regular arrays.</td>
    </tr>
    <tr>
      <td style="background-color: #5b615c; padding: 10px; border: 1px solid #ddd;">Rich Metadata</td>
      <td style="background-color: #A5D6A7; padding: 10px; border: 1px solid #ddd;">1</td>
      <td style="background-color: #A5D6A7; padding: 10px; border: 1px solid #ddd;">1</td>
      <td style="background-color: #FFF59D; padding: 10px; border: 1px solid #ddd;">2</td>
      <td style="padding: 10px; border: 1px solid #ddd;">mGeoTIFF only supports JSON-encoded metadata.</td>
    </tr>
  </tbody>
</table>
*1: Excellent, 2: Okay, 3: Poor* 


## Proof of Concept

We successfully transformed a massive <b style="color: #D32F2F;">4 Terabyte</b> dataset of <b style="color: #D32F2F;">200,000</b> ZARR files into a streamlined, **cloud-native** <b style="color: #388E3C;">700 GB</b> dataset of just few files. This optimization was achieved using TACO and Temporal GeoTIFFs.

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

### Writing a Multidimensional GeoTIFF

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
    data_r = src.read()

# 5. Convert the data back to an xarray DataArray
datacube_r = xr.DataArray(
    data=data_r,
    dims=src.md_meta["md:dimensions"],
    coords=src.md_meta["md:coordinates"],
    attrs=src.md_meta["md:attributes"]
)

# 6. Assert that the data is the same
assert np.allclose(datacube, datacube_r)
```

### Writing a Temporal GeoTIFF

A reproducible and **real-world example** of a Temporal GeoTIFF: using this format reduces file size by 60% compared 
to storing different time steps in separate files.

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
    
    # Get the snipet of the Sentinel-2 image
    s2_str = row.read(0).read(0)

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
    'zstd_level': 22,
    'predictor': 2,
    'tiled': True,
    'num_threads': 10,
    'interleave': 'pixel',
    'crs': row["stac:crs"].iloc[0],
    'transform': mrio.Affine.from_gdal(*row["stac:geotransform"].iloc[0]),
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

# 7. Read the data
with mrio.open("temporal_stack.tif") as src:
    data_r = src.read()
    attributes = src.attributes()
    temporal_stack = xr.DataArray(
        data=data_r,
        dims=attributes["mrio:dimensions"],
        coords=attributes["mrio:coordinates"],
        attrs=attributes["mrio:attributes"]
    )
```

### Documentation

See documentation for more details: [https://tacofoundation.github.io/mrio/](https://tacofoundation.github.io/mrio/)

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Changes

See the [CHANGELOG](CHANGELOG.md) file for details.
