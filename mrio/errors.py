"""Custom exceptions for the MRIO package."""

import json

from rasterio.errors import RasterioIOError


class MRIOError(Exception):
    """Base exception for MRIO package."""


class ValidationError(MRIOError):
    """Raised when data validation fails."""

# Error for Slice Transformer (slice_transformer.py)
class SliceEllipsisError(IndexError):
    def __init__(self):
        super().__init__("Only one ellipsis (...) allowed in indexing expression")


class SliceEllipsisTooManyIndicesError(IndexError):
    def __init__(self, ndim):
        super().__init__(f"Too many indices for {ndim}D array")


class SliceUnsupportedTypeError(TypeError):
    def __init__(self, key):
        super().__init__(f"Unsupported key type: {type(key)}")


# Error for Earth Engine API (fields.py)
class FieldsCoordinateMDPattern(ValidationError):
    def __init__(self):
        super().__init__("Coordinates keys must match the values in parentheses")


class FieldsCoordinateListError(ValidationError):
    def __init__(self):
        super().__init__("All coordinate values must be lists")


class FieldsPatternSeparatorError(ValidationError):
    def __init__(self):
        super().__init__("Pattern must contain ' -> ' separator")


class FieldsPatternError(ValidationError):
    def __init__(self):
        super().__init__("Invalid pattern format")


class FieldsEmptyCoordinatesError(ValidationError):
    def __init__(self):
        super().__init__("Coordinates cannot be empty")


class FieldsPatternParseError(ValueError):
    def __init__(self):
        super().__init__("Pattern must be validated before parsing")


class FieldsCoordinateMismatchError(ValueError):
    def __init__(self):
        super().__init__(
            "Coordinate names after '->' must match in length with names before '->'"
        )


class FieldsMissingFieldsError(ValueError):
    def __init__(self, missing_fields):
        super().__init__(f"Mandatory fields missing: {', '.join(missing_fields)}")

class FieldsBlockZSizeError(ValueError):
    def __init__(self):
        super().__init__("BlockZSize must be a positive integer")


# Error for Earth Engine API (earthengine_api.py)
class EEInvalidDateRangeError(ValueError):
    """Raised when date range validation fails."""

    def __init__(self):
        super().__init__("Invalid date range: start date must be before end date")


class EEInvalidBandsError(ValueError):
    """Raised when band selection is invalid."""

    def __init__(self, available_bands):
        super().__init__(f"Invalid bands. Available bands: {available_bands}")


class EENoDateError(ValueError):
    """Raised when no dates are found in specified range."""

    def __init__(self):
        super().__init__("No dates found within specified range")


class EEMissingBlockSizeError(ValueError):
    """Raised when dataset profile lacks block size info."""

    def __init__(self):
        super().__init__("Dataset profile missing block size information")


class EEInvalidDateFormatError(ValueError):
    """Raised when date format is invalid."""

    def __init__(self, date_error):
        super().__init__(f"Invalid date format. Use YYYY-MM-DD: {date_error}")


# Error for reading (readers.py)
class ReadersFailedToOpenError(RasterioIOError):
    def __init__(self, file_path, e):
        super().__init__(f"Failed to open {file_path}: {e}")


class ReadersInstallxarrayError(ImportError):
    def __init__(self):
        super().__init__(
            "xarray is required for this operation.\nInstall it via: pip install xarray"
        )


# Error for temporal_utils.py
class TemporalCRSError(ValueError):
    def __init__(self, idx, profile, base):
        super().__init__(
            f"CRS mismatch: Image {idx} has {profile['crs']}, expected {base['crs']}"
        )


class TemporalTransformError(ValueError):
    def __init__(self, idx):
        super().__init__(f"Transform mismatch: Image {idx} has different transform")


class TemporalDimensionError(ValueError):
    def __init__(self, idx, profile, base):
        super().__init__(
            f"Dimension mismatch: Image {idx} is {profile['width']}x{profile['height']}, "
            f"expected {base['width']}x{base['height']}"
        )


class TemporalFileDateMismatchError(ValueError):
    def __init__(self):
        super().__init__("Number of files must match number of start dates")


class TemporalEndDateMismatchError(ValueError):
    def __init__(self):
        super().__init__("Number of end dates must match number of files")


# Error for validators.py
class ValidatorsFailedToOpenError(RasterioIOError):
    def __init__(self, file_path, e):
        super().__init__(f"Failed to open file: {file_path}: {e}")


class ValidatorsInvalidMetadataError(json.JSONDecodeError):
    def __init__(self, e):
        super().__init__(f"Invalid metadata JSON: {e}")


# Error for writers.py
class WritersFailedToOpenError(RasterioIOError):
    def __init__(self, file_path, e):
        super().__init__(f"Failed to open {file_path} for writing: {e}")


class WritersUnsupportedDataTypeError(ValueError):
    def __init__(self, data_type):
        super().__init__(f"Unsupported data type: {data_type}")


# Error for chunk_reader.py
class ChunkReaderNoQueryError(MRIOError):
    def __init__(self):
        super().__init__("No query has been executed yet")


class ChunkReaderNoDataError(MRIOError):
    def __init__(self):
        super().__init__("No data has been read yet")


class ChunkReaderInvalidFilterError(ValueError):
    def __init__(self):
        super().__init__("Filter criteria must be slice, int, list, or tuple")


class ChunkReaderInvalidConditionError(ValueError):
    def __init__(self, cond):
        super().__init__(f"Unsupported condition type: {type(cond)}")

class ChunkReaderInvalidFilterCriteriaError(ValueError):
    def __init__(self):
        super().__init__("Filter criteria must be a slice object")
