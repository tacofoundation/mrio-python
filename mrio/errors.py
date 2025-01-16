class MRIOError(Exception):
    """Base exception for MRIO package."""


class MetadataError(MRIOError):
    """Raised when metadata validation fails."""


class ValidationError(MRIOError):
    """Raised when data validation fails."""


class MRIOIOError(MRIOError, IOError):
    """Raised when I/O operations fail."""
