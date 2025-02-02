"""A module for temporal utilities."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
from typing import List, Union, Optional, Dict, Any, Literal
from os import PathLike

import numpy as np
import rasterio as rio
from mrio.types import PathLike
from mrio.writers import DatasetWriter

from mrio.env_options import MRIOConfig
from mrio.readers import DatasetReader


def merge_profiles(base: dict, updates: dict) -> dict:
    """Deep merge two profiles with special handling for md:attributes.

    Args:
        base: Base profile dictionary
        updates: Updates to apply

    Returns:
        dict: Merged profile
    """
    result = base.copy()

    for key, value in updates.items():
        if key == "md:attributes" and key in result:
            # Merge md:attributes dictionaries
            result[key] = {**result[key], **value}
        else:
            result[key] = value

    return result


def validate_cog_spatial_consistency(profiles: List[dict]) -> dict:
    """
    Validate spatial consistency across all COG files and return base profile.

    Checks that all images have the same CRS, transform, width and height.
    Returns the full profile from the first image if all checks pass.

    Args:
        profiles: List of rasterio profiles from input images

    Returns:
        dict: Full profile from first image

    Raises:
        ValueError: If spatial properties don't match across images
    """
    base = profiles[0]

    for i, profile in enumerate(profiles[1:], 1):
        # Check CRS
        if base["crs"] != profile["crs"]:
            raise ValueError(f"CRS mismatch: Image {i} has {profile['crs']}, expected {base['crs']}")

        # Check transform with numpy allclose for float comparison
        if not np.allclose(base["transform"], profile["transform"]):
            raise ValueError(f"Transform mismatch: Image {i} has different transform")

        # Check dimensions
        if base["width"] != profile["width"] or base["height"] != profile["height"]:
            raise ValueError(
                f"Dimension mismatch: Image {i} is {profile['width']}x{profile['height']}, "
                f"expected {base['width']}x{base['height']}"
            )
    return base


def read_cog_file(file: PathLike) -> tuple[np.ndarray, dict]:
    """Read a COG file and return data and profile."""
    with rio.open(file) as src:
        return src.read(), src.profile


def create_metadata_profile(
    base_profile: dict,
    dataset: np.ndarray,
    files: List[PathLike],
    start_dates: List[Union[datetime, int]],
    end_dates: Optional[List[Union[datetime, int]]] = None,
    blocksize: int = 64,
) -> dict:
    """Create metadata profile using full base profile."""

    # Convert dates to required formats
    start_dates_int = [int(x.timestamp()) if isinstance(x, datetime) else x for x in start_dates]
    start_dates_str = [x.strftime("%Y%m%d") if isinstance(x, datetime) else str(x) for x in start_dates]

    # Start with complete base profile
    profile = base_profile.copy()

    # Update with temporal stack specific metadata
    profile.update({
        "blocksize": blocksize,
        "md:pattern": "time band y x -> (band time) y x",
        "md:coordinates": {"time": start_dates_str, "band": [f"B{i:02d}" for i in range(1, dataset.shape[1] + 1)]},
        "md:attributes": {"md:id": [Path(x).stem for x in files], "md:time_start": start_dates_int},
    })

    # Add end dates if provided
    if end_dates:
        end_dates_int = [int(x.timestamp()) if isinstance(x, datetime) else x for x in end_dates]
        profile["md:attributes"]["md:time_end"] = end_dates_int

    return profile


def stack_temporal(
    cog_files: List[PathLike],
    output_file: PathLike,
    start_date: Union[List[int], List[datetime]],
    blocksize: Optional[int] = 64,
    end_date: Optional[Union[List[int], List[datetime]]] = None,
    n_threads: int = 4,
    **kwargs: Dict[str, Any],
) -> Path:
    """
    Stack multiple COG files into a single multitemporal COG file.

    Efficiently combines multiple Cloud Optimized GeoTIFFs into a single temporal stack,
    with parallel file reading and proper metadata handling. Validates spatial consistency
    across all input images.

    Args:
        cog_files: List of paths to input COG files
        output_file: Path for output temporal COG
        start_date: List of start dates for each input file
        chunk_size: Block size for COG tiling (default: 64)
        end_date: Optional list of end dates for each file
        n_threads: Number of parallel threads for reading (default: 4)
        **kwargs: Additional profile parameters

    Returns:
        Path: Path to created output file

    Raises:
        ValueError: If input lists have mismatched lengths or spatial properties don't match
        RuntimeError: If file reading fails
    """
    # Validate inputs
    if len(cog_files) != len(start_date):
        raise ValueError("Number of files must match number of start dates")
    if end_date and len(end_date) != len(cog_files):
        raise ValueError("Number of end dates must match number of files")

    # Read files in parallel
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(executor.map(read_cog_file, cog_files))

    # Separate data and profiles
    data_arrays, profiles = zip(*results)

    # Validate spatial consistency and get base profile
    base_profile = validate_cog_spatial_consistency(profiles)

    # Stack data
    dataset = np.array(data_arrays)

    # Create metadata profile
    profile = create_metadata_profile(base_profile, dataset, cog_files, start_date, end_date, blocksize)

    # Merge with kwargs, preserving md:attributes
    cog_profile = merge_profiles(profile, kwargs)

    # Update driver
    cog_profile["driver"] = "COG"

    # Handle block size parameters
    if "blockxsize" in cog_profile:
        cog_profile["blocksize"] = cog_profile.pop("blockxsize")
        cog_profile.pop("blockysize", None)  # Remove if exists

    # Remove tiled parameter if exists
    cog_profile.pop("tiled", None)

    # Change interleave to tile
    cog_profile["interleave"] = "TILE"

    with DatasetWriter(output_file, **cog_profile) as dst:
        dst.write(dataset)

    return Path(output_file)


def unstack_temporal(
    filepath: PathLike,
    output_dir: PathLike,
    read_env_options: Union[Literal["mrio", "default"], Dict[str, str]] = "mrio",
    n_threads: int = 4,
    **kwargs: Dict[str, Any],
) -> List[Path]:
    """Unstack a temporal Cloud Optimized GeoTIFF (COG) file into multiple single-timestep COG files.

    Args:
        filepath: Path or HTTP URL to the input temporal COG file.
        output_dir: Directory where the unstacked COG files will be saved.
        read_env_options: Environment configuration for MRIO. Either "mrio", "default",
            or a dictionary of custom environment variables.
        n_threads: Number of parallel threads for writing files. Default is 4.
        **kwargs: Additional parameters passed to rasterio when creating COG files.

    Returns:
        List[Path]: Sorted list of paths to the created unstacked COG files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with MRIOConfig.get_env(read_env_options):
        # Read dataset metadata once
        with DatasetReader(filepath, engine="numpy") as dataset:
            metadata = dataset.md_meta
            attributes = metadata["md:attributes"]

            filenames = attributes["md:id"]
            time_start = attributes["md:time_start"]
            time_end = attributes.get("md:time_end")

            # Extract other metadata attributes
            other_attributes = {
                key: attributes[key]
                for key in attributes.keys()
                if key not in ["md:id", "md:time_start", "md:time_end"]
            }

            # Prepare COG profile
            cog_profile = {
                "driver": "COG",
                "blocksize": dataset.block_shapes[0],
                "interleave": "PIXEL",
                "crs": dataset.crs,
                "transform": dataset.transform,
                "width": dataset.width,
                "height": dataset.height,
                "count": dataset.shape[1],
                "dtype": dataset.dtype,
                "nodata": dataset.nodata,
                "compress": dataset.compression.value,
                **kwargs,
            }

        def process_timestep(idx: int) -> Path:
            """Process a single timestep from the temporal COG file."""
            with DatasetReader(filepath, engine="numpy") as dataset:
                data = dataset[idx].squeeze()

                # Prepare metadata for this timestep
                md_meta = {
                    "md:attributes": {"md:id": filenames[idx], "md:time_start": time_start[idx], **other_attributes}
                }

                if time_end:
                    md_meta["md:attributes"]["md:time_end"] = time_end[idx]

                output_path = output_dir / filenames[idx]
                with rio.open(output_path, "w", **cog_profile) as dst:
                    dst.write(data)
                    dst.update_tags(MD_META=json.dumps(md_meta))

                return output_path

        # Process timesteps in parallel
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            output_files = list(executor.map(process_timestep, range(len(time_start))))

        return sorted(output_files)
