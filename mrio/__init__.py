import importlib
import pkgutil
import warnings

import rasterio
from rasterio.errors import RasterioDeprecationWarning

warnings.filterwarnings("ignore", category=RasterioDeprecationWarning)

# Import all public attributes from the main rasterio module
# TODO: I don't like this, but idk how to do it better 
globals().update(
    {
        name: getattr(rasterio, name)
        for name in dir(rasterio)
        if not name.startswith("_")  # Exclude private attributes
    }
)

# Dynamically discover and import all submodules of rasterio
# TODO: I don't like this, but idk how to do it better 
for _, submodule_name, is_pkg in pkgutil.walk_packages(
    rasterio.__path__, prefix="rasterio."
):
    try:
        submodule = importlib.import_module(submodule_name)
        globals()[submodule_name.split(".")[-1]] = submodule
    except ImportError as e:
        print(f"Failed to import {submodule_name}: {e}")


# Import the main I/O MRIO function
from mrio.main import open
from mrio.validators import is_mgeotiff, is_tgeotiff

__version__ = "0.0.9"
