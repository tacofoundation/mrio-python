""" HTTP and caching configuration options for mrio.

This module provides configuration options for rasterio/GDAL to optimize
HTTP requests and caching. The configuration options are used to create 
a rasterio environment that can be used as a context manager to open 
and read mCOG files.
"""

from typing import ClassVar, Literal

from rasterio.env import Env


class MRIOConfig:
    """
    Provides predefined configurations for optimizing rasterio/GDAL environments.
    
    Contains two class-level configurations:
    - DEFAULT_CONFIG: GDAL Conservative settings for general-purpose raster operations
    - MRIO_CONFIG: Enhanced configuration optimized for multi-threaded processing
        and better mCOG performance.
    """
    
    # Base configuration balancing compatibility and performance
    DEFAULT_CONFIG: ClassVar[dict[str, str | int]] = {
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "GDAL_DISABLE_READDIR_ON_OPEN": "FALSE",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "",
        "GDAL_INGESTED_BYTES_AT_OPEN": 16384,
        "GDAL_CACHEMAX": 64,
        "CPL_VSIL_CURL_CACHE_SIZE": 16777216,
        "CPL_VSIL_CURL_CHUNK_SIZE": 16384,
        "VSI_CACHE": "FALSE",
        "VSI_CACHE_SIZE": 26214400,
        "GDAL_BAND_BLOCK_CACHE": "AUTO",
    }

    # Optimized configuration for high-performance multi-threaded reading
    MRIO_CONFIG: ClassVar[dict[str, str | int]] = {
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
        "GDAL_INGESTED_BYTES_AT_OPEN": 393216,  # 384 KB  (optimized for large mCOG headers)
        "GDAL_CACHEMAX": 512,
        "CPL_VSIL_CURL_CACHE_SIZE": 167772160,
        "CPL_VSIL_CURL_CHUNK_SIZE": 10485760,
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": 10485760,
        "GDAL_BAND_BLOCK_CACHE": "HASHSET",
        "PROJ_NETWORK": "ON",  # Enable PROJ network for geodetic transformations
    }

    @staticmethod
    def get_env(config: Literal["default", "mrio"] | dict[str, str] = "mrio") -> Env:
        """
        Create a configured rasterio environment with optimized GDAL settings.

        Args:
            config: Configuration selection. Can be:
                - "mrio": Use MRIO_CONFIG (default)
                - "default": Use DEFAULT_CONFIG
                - Custom dictionary: User-provided GDAL options

        Returns:
            Configured rasterio environment context manager

        Examples:
            Basic usage with default MRIO configuration:
            >>> with MRIOConfig.get_env() as env:
            >>>     with rasterio.open('large_cog.tif') as src:
            >>>         print(src.profile)

            Using conservative defaults:
            >>> with MRIOConfig.get_env('default') as env:
            >>>     # Perform raster operations
        """
        if config == "mrio":
            settings = MRIOConfig.MRIO_CONFIG
        elif config == "default":
            settings = MRIOConfig.DEFAULT_CONFIG
        else:
            settings = config
        return Env(**settings)

    @staticmethod
    def print_config(config: dict) -> None:
        """
        Display configuration settings in a standardized format.

        Args:
            config: Configuration dictionary to display

        Example:
            >>> MRIOConfig.print_config(MRIOConfig.MRIO_CONFIG)
            >>> custom_config = {"GDAL_CACHEMAX": 256}
            >>> MRIOConfig.print_config(custom_config)
        """
        print("\nRasterio/GDAL Configuration:")
        for key, value in sorted(config.items()):
            print(f"{key}: {value}")
