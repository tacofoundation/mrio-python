from typing import ClassVar, Literal

from rasterio.env import Env


class MRIOConfig:
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

    MRIO_CONFIG: ClassVar[dict[str, str | int]] = {
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
        "GDAL_INGESTED_BYTES_AT_OPEN": 393216,  # 384 KB
        "GDAL_CACHEMAX": 512,
        "CPL_VSIL_CURL_CACHE_SIZE": 167772160,
        "CPL_VSIL_CURL_CHUNK_SIZE": 10485760,
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": 10485760,
        "GDAL_BAND_BLOCK_CACHE": "HASHSET",
        "PROJ_NETWORK": "ON",
    }

    @staticmethod
    def get_env(config: Literal["default", "mrio"] | dict[str, str] = "mrio") -> Env:
        """
        Get a rasterio environment with the specified configuration.

        Args:
            config (Dict, optional): Custom configuration dictionary.
                                   If None, uses MRIO_CONFIG.

        Returns:
            rasterio.env.Env: Configured rasterio environment

        Example:
            # Use as a context manager
            with RioConfig.get_env() as env:
                with rasterio.open('my_file.tif') as src:
                    # Do something with src
                    pass
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
        """Print configuration settings in a readable format."""
        print("\nRasterio/GDAL Configuration:")
        for key, value in sorted(config.items()):
            print(f"{key}: {value}")
