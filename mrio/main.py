import json
import math
from itertools import product
from pathlib import Path
from typing import Any, Optional, List

import rasterio as rio
import numpy as np
from einops import rearrange
from mrio.datamodel import MRIOFields


class DatasetReader:
    def __init__(self, file_path: Path, mode: str, *args, **kwargs):
        """
        Initialize DatasetReader for reading/writing custom data formats.

        Args:
            file_path (Path): Path to the dataset file.
            mode (str): Mode ('r' for read, 'w' for write).
            *args: Additional arguments for rasterio.
            **kwargs: Additional keyword arguments for rasterio.
        """
        self.file_path = file_path
        self.mode = mode.lower()
        self.args = args
        self.kwargs = kwargs        

        # Initialize mode-specific settings
        if self.mode == "w":
            self._initialize_write_mode()
        elif self.mode != "r":
            raise ValueError("Invalid mode. Use 'r' for read or 'w' for write.")

        # Open the file
        self._file = rio.open(self.file_path, self.mode, *self.args, **self.kwargs)

        # Initialize attributes for read mode
        if self.mode == "r":
            self._initialize_read_mode()


    def _initialize_write_mode(self):
        """Validate and process metadata for write mode."""
        md_kwargs_dict = {
            k.split("md:")[1]: self.kwargs.pop(k)
            for k in list(self.kwargs)
            if k.startswith("md:")
        }
        self.md_kwargs = MRIOFields(**md_kwargs_dict)
        self.kwargs["count"] = math.prod(
            len(values) for values in self.md_kwargs.coordinates.values()
        )

    def _initialize_read_mode(self):
        """Load metadata and attributes for read mode."""
        self.profile = self._file.profile
        self.meta = self._file.meta
        self.width = self.profile.get("width")
        self.height = self.profile.get("height")
        self.crs = self.profile.get("crs")
        self.transform = self.profile.get("transform")
        self.count = self.profile.get("count")
        self.indexes = self._file.indexes
        self.window = self._file.window
        self.bounds = self._file.bounds
        self.res = self._file.res
        self.shape = self._file.shape
        self.dtypes = self._file.dtypes
        self.nodata = self._file.nodata

        # Load the multi-dimensional metadata
        self.md_meta = self._get_md_metadata()

    def __enter__(self):
        """Enter the runtime context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context and close the file."""
        self.close()

    def close(self):
        """Ensure the file is closed."""
        if self._file and not self._file.closed:
            self._file.close()
        

    def read(self, *args, **kwargs) -> Any:
        """Read data from the custom file format."""
        if self.mode != "r":
            raise ValueError("File must be opened in read mode.")
        return self._read_custom_data(*args, **kwargs)

    def write(self, data: Any):
        """Write data to the custom file format."""
        if self.mode != "w":
            raise ValueError("File must be opened in write mode.")
        self._write_custom_data(data)

    def __del__(self):
        """Ensure the file is closed when the object is deleted."""
        self.close()        

    def _read_custom_data(self, *args, **kwargs) -> Any:
        """Internal method to read custom data from the file."""
        
        # If no metadata is present, treat the file as a standard GeoTIFF                
        if not self.md_meta:            
            return self._file.read(*args, **kwargs)

        # Read raw data and rearrange based on metadata
        raw_data = self._file.read(*args, **kwargs)        
        return rearrange(raw_data, self.md_meta["md:pattern"], **self.md_meta["md:coordinates_len"])

    def _write_custom_data(self, data: Any):
        """Internal method to write custom data to the file."""

        # Handle non-multidimensional data
        if not self.md_kwargs.pattern or not self.md_kwargs.coordinates:
            self._file.write(data)
            return
        
        # Rearrange data according to the pattern
        image: np.ndarray = rearrange(data, self.md_kwargs.pattern)

        # Generate unique band identifiers
        band_unique_identifiers: List[str] = [
            "__".join(combination) 
            for combination in product(
                *[self.md_kwargs.coordinates[band] for band in self.md_kwargs._in_parentheses]
            )
        ] 

        # Create GLOBAL METADATA for the file
        md_metadata = json.dumps({
            "md:dimensions": self.md_kwargs._before_arrow,
            "md:coordinates": self.md_kwargs.coordinates,
            "md:coordinates_len": {
                band: len(coords) for band, coords in self.md_kwargs.coordinates.items()
                if band in self.md_kwargs._in_parentheses
            },
            "md:attributes": self.md_kwargs.attributes,
            "md:pattern": self.md_kwargs._inv_pattern,
        })

        # Write bands and set descriptions
        for i, (band_data, band_id) in enumerate(zip(image, band_unique_identifiers), start=1):
            self._file.write(band_data, i)
            self._file.set_band_description(i, band_id)

        # Update file metadata
        self._file.update_tags(MD_METADATA=md_metadata)

    def _get_md_metadata(self) -> Optional[dict]:
        """Retrieve multi-dimensional metadata."""

        # Retrieve metadata from the file tags
        metadata = self._file.tags().get("MD_METADATA")
        if not metadata:
            return None

        # If metadata is present, parse and return it
        return json.loads(metadata)
    
    def tags(self) -> dict:
        """Retrieve all tags from the file."""
        return self._file.tags()
    
    def __repr__(self):
        if self._file.closed:
            return f"<close DatasetReader name='{self.file_path}' mode='{self.mode}'>"
        return f"<open DatasetReader name='{self.file_path}' mode='{self.mode}'>"
    
    def __str__(self):
        return repr(self)

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
