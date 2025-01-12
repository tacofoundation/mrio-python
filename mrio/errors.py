class MRIOError(Exception):
    """Base exception for MRIO package."""

    pass


class MetadataError(MRIOError):
    """Raised when metadata validation fails."""

    pass


class ValidationError(MRIOError):
    """Raised when data validation fails."""

    pass


class MRIOIOError(MRIOError, IOError):
    """Raised when I/O operations fail."""

    pass
