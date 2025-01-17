
### Validation

The `mrio` library includes two functions to validate if a file is a mGeoTIFF or a tGeoTIFF.

```python
import mrio

# If the file is a mGeoTIFF, it will return True
mrio.is_mgeotiff("image.tif")

# If the file is a tGeoTIFF, it will return True
mrio.is_tgeotiff("temporal_stack.tif")
```