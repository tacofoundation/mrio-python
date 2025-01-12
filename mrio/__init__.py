import importlib
import pkgutil
import warnings
from pathlib import Path

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
from mrio.dataset import DatasetReader
from mrio.validators import is_mgeotiff, is_tgeotiff
from importlib.metadata import version


__version__ = version("mrio")


def open(file_path: Path, mode: str = "r", *args, **kwargs) -> DatasetReader:
    """
    Open a DatasetReader object.

    Args:
        file_path (Path): Path to the dataset file.
        mode (str): Mode ('r' for read, 'w' for write).
        *args: Additional arguments for rasterio.
        **kwargs: Additional keyword arguments for rasterio.

    Returns:
        DatasetReader: An instance of DatasetReader.
    """

    return DatasetReader(file_path, mode, *args, **kwargs)
