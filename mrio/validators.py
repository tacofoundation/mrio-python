import rasterio as rio
import json

def check_metadata(path: str, required_fields: list, required_attributes: list = None) -> bool:
    """
    Check if a GeoTIFF file contains valid MD_METADATA with specified fields and attributes.

    Args:
        path (str): Path to the GeoTIFF file.
        required_fields (list): List of fields required in the top-level metadata.
        required_attributes (list, optional): List of attributes required within `md:attributes`.

    Returns:
        bool: True if the metadata is valid and contains all required fields and attributes, False otherwise.
    """
    try:
        with rio.open(path) as src:
            tags = src.tags()
    except Exception as e:
        print(f"Failed to open file: {e}")
        return False

    # Check for MD_METADATA
    if "MD_METADATA" not in tags:
        print("MD_METADATA not found")
        return False

    # Validate JSON structure of MD_METADATA
    try:
        metadata = json.loads(tags["MD_METADATA"])
    except json.JSONDecodeError:
        print("An error occurred while parsing MD_METADATA")
        return False

    # Check for required fields
    if not all(field in metadata for field in required_fields):
        print(f"Missing required fields in MD_METADATA: {required_fields}")
        return False

    # Check for required attributes if specified
    if required_attributes:
        attributes = metadata.get("md:attributes", {})
        if not all(attr in attributes for attr in required_attributes):
            print(f"Missing required attributes in md:attributes: {required_attributes}")
            return False

    return True

def is_mgeotiff(path: str) -> bool:
    """
    Check if a GeoTIFF file is a valid MGeoTIFF.

    Args:
        path (str): Path to the GeoTIFF file.

    Returns:
        bool: True if the file is a valid MGeoTIFF, False otherwise.
    """
    required_fields = ["md:pattern", "md:coordinates"]
    return check_metadata(path, required_fields)

def is_tgeotiff(path: str) -> bool:
    """
    Check if a GeoTIFF file is a valid TGeoTIFF.

    Args:
        path (str): Path to the GeoTIFF file.

    Returns:
        bool: True if the file is a valid TGeoTIFF, False otherwise.
    """
    required_fields = ["md:pattern", "md:coordinates"]
    required_attributes = ["md:time_start", "md:id"]
    return check_metadata(path, required_fields, required_attributes)