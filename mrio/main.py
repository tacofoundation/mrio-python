import json
import math
import re
from itertools import product
from pathlib import Path
from typing import Any, Optional

import rasterio as rio
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

        # Handle MRIO-specific arguments when in write mode
        if self.mode == "w":
            mrio_kwargs = {
                k: kwargs.pop(k) for k in list(kwargs) if k.startswith("mrio:")
            }
            # Validate and store MRIO fields
            self.mrio_kwargs = MRIOFields(**mrio_kwargs)

            # Extract MRIO fields
            self.mrio_strategy = self.mrio_kwargs.strategy
            self.mrio_coordinates = self.mrio_kwargs.coordinates
            self.mrio_attributes = self.mrio_kwargs.attributes
            self.kwargs["count"] = math.prod(
                len(v) for v in self.mrio_coordinates.values()
            )
        else:
            self.mrio_strategy = None
            self.mrio_coordinates = None
            self.mrio_attributes = None

        self._file = None
        self.profile = None
        self.meta = None

    def __enter__(self):
        """Enter the runtime context and open the file."""
        self._file = rio.open(self.file_path, self.mode, *self.args, **self.kwargs)
        if self.mode == "r":
            self.profile = self._file.profile.copy()
            self.meta = self._file.meta.copy()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context and close the file."""
        if self._file:
            self._file.close()

    def read(self) -> Any:
        """Reads data from the custom file format."""
        if self.mode != "r":
            raise ValueError("File must be opened in read mode.")
        return self._read_custom_data()

    def write(self, data: Any):
        """Writes data to the custom file format."""
        if self.mode != "w":
            raise ValueError("File must be opened in write mode.")
        self._write_custom_data(data)

    def _read_custom_data(self) -> Any:
        """Internal method to read custom data from the file."""
        all_tags = self._file.tags()
        mrio_pattern = all_tags.get("MULTIDIMENSIONAL_PATTERN")
        mrio_coordinates = all_tags.get("MULTIDIMENSIONAL_COORDINATES")
        raw_data = self._file.read()

        if not mrio_pattern or not mrio_coordinates:
            return raw_data

        return rearrange(raw_data, mrio_pattern, **json.loads(mrio_coordinates))

    def _write_custom_data(self, data: Any):
        """Internal method to write custom data to the file."""
        if not self.mrio_strategy or not self.mrio_coordinates:
            self._file.write(data)
            return

        image = rearrange(data, self.mrio_strategy)
        read_strategy = " -> ".join(
            map(str.strip, self.mrio_strategy.split("->")[::-1])
        )

        # Validate band names
        band_names = re.search(r"\((.*?)\)", self.mrio_strategy).group(1).split()
        missing_bands = [
            band for band in band_names if band not in self.mrio_coordinates
        ]
        if missing_bands:
            raise ValueError(
                f"Missing 'mrio:coordinates' for bands: {', '.join(missing_bands)}"
            )

        # Create band identifiers
        band_combinations = list(
            product(*(self.mrio_coordinates[band] for band in band_names))
        )
        band_identifiers = ["__".join(combination) for combination in band_combinations]

        coordinates_json = json.dumps(
            {band: len(self.mrio_coordinates[band]) for band in band_names}
        )
        attributes_json = (
            json.dumps(self.mrio_attributes) if self.mrio_attributes else None
        )

        for i, band_data in enumerate(image, start=1):
            self._file.write(band_data, i)
            self._file.set_band_description(i, band_identifiers[i - 1])
            self._file.update_tags(
                MULTIDIMENSIONAL_PATTERN=read_strategy,
                MULTIDIMENSIONAL_COORDINATES=coordinates_json,
            )
            if attributes_json:
                self._file.update_tags(MULTIDIMENSIONAL_ATTRIBUTES=attributes_json)

    def attributes(self) -> Optional[dict]:
        """Returns the attributes of the dataset."""
        mrio_attributes = self._file.tags().get("MULTIDIMENSIONAL_ATTRIBUTES")
        return json.loads(mrio_attributes) if mrio_attributes else None


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
    if mode == "r" and not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return DatasetReader(file_path, mode, *args, **kwargs)
