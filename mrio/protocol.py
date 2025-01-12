from typing import Protocol, runtime_checkable

from mrio.typing import ArrayLike


@runtime_checkable
class DatasetProtocol(Protocol):
    """Protocol for dataset operations."""

    def read(self) -> ArrayLike: ...
    def write(self, data: ArrayLike) -> None: ...
    def close(self) -> None: ...

    @property
    def closed(self) -> bool: ...
