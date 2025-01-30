""" A module for temporal utilities. """
from typing import List, Optional

from mrio.types import PathLike


def unstack_temporal(filepath: PathLike, output_dir: PathLike) -> None:
    """Unstacks a temporal COG file into multiple COG files, one for each time step.

    Args:
        filepath: Path to the input temporal COG file.
        output_dir: Path to the output directory where files will be saved.

    Returns:
        A list of unstacked file paths. The files names are those specified in the
        `md:id` attribute of the input file.

    Examples:
        import mrio

        input_file = 'path/to/input.tif'
        output_dir = 'path/to/output'

        mrio.temporal.unstack(input_file, output_dir)
        # >>> ['path/to/output/file1.tif', 'path/to/output/file2.tif', ...]
    """
    pass


def stack_temporal(filepaths: List[PathLike], output_file: PathLike, interleave: Optional[str] = "tile", **kwargs: dict) -> None:
    """Stacks multiple COG files into a single temporal COG file.

    Args:
        filepaths: List of paths to the input COG files.
        output_file: Path to the output temporal COG file.
        interleave: Interleave pattern for the output file. Can be 'band', 'tile', or 'pixel'.
        **kwargs: Additional keyword arguments passed to the writer.
    """
    pass
