import rasterio
import pkgutil
import importlib
import warnings

from rasterio.errors import RasterioDeprecationWarning

warnings.filterwarnings("ignore", category=RasterioDeprecationWarning)

# Import all public attributes from the main rasterio module
globals().update({
    name: getattr(rasterio, name)
    for name in dir(rasterio)
    if not name.startswith('_')  # Exclude private attributes
})

# Dynamically discover and import all submodules of rasterio
for _, submodule_name, is_pkg in pkgutil.walk_packages(rasterio.__path__, prefix='rasterio.'):    
    try:        
        submodule = importlib.import_module(submodule_name)
        globals()[submodule_name.split(".")[-1]] = submodule
    except ImportError as e:
        print(f"Failed to import {submodule_name}: {e}")


# Import the main I/O MRIO function
from mrio.main import open
