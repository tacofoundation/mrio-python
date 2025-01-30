#!/usr/bin/env python3
import struct

# Minimal constants for TIFF
TIFF_BIGENDIAN = b'MM'
TIFF_LITTLEENDIAN = b'II'
TIFF_MAGIC = 42    # Classic TIFF

SUBFILE_TYPE_TAG = 254
IMAGE_WIDTH_TAG = 256
IMAGE_LENGTH_TAG = 257
BITS_PER_SAMPLE_TAG = 258
COMPRESSION_TAG = 259
PHOTOMETRIC_TAG = 262
STRIP_OFFSETS_TAG = 324   # same ID as tile offsets in a tiled TIFF
STRIP_BYTE_COUNTS_TAG = 325  # same ID as tile byte counts
SAMPLES_PER_PIXEL_TAG = 277
PLANAR_CONFIG_TAG = 284
TILE_WIDTH_TAG = 322
TILE_LENGTH_TAG = 323



# For simplicity, handle only these basic TIFF datatypes:
TYPE_BYTE = 1
TYPE_SHORT = 3
TYPE_LONG = 4

# Minimal "ghost" area for a COG; ensures that the IFD block is at the front:
# Adjust the size in the string if you want to preserve exact alignment.
GHOST = (
    b"GDAL_STRUCTURAL_METADATA_SIZE=000140 bytes\n"
    b"LAYOUT=IFDS_BEFORE_DATA\n"
    b"BLOCK_ORDER=ROW_MAJOR\n"
    b"BLOCK_LEADER=SIZE_AS_UINT4\n"
    b"BLOCK_TRAILER=LAST_4_BYTES_REPEATED\n"
    b"KNOWN_INCOMPATIBLE_EDITION=NO\n  "
    # The trailing spaces ensure correct alignment. Typically 2 spaces at end.
)

class IFDEntry:
    """
    Represents a single IFD entry (tag, type, count, value_offset).
    We store the raw 'value_offset' as read from the file. If 'count * size_of_type' > 4,
    the actual data resides at that offset, otherwise the data might be inlined.
    """
    __slots__ = ('count', 'tag', 'type', 'value_offset')

    def __init__(self, tag, ttype, count, value_offset):
        self.tag = tag
        self.type = ttype
        self.count = count
        self.value_offset = value_offset

class TiffIFD:
    """
    Represents a parsed IFD:
      - width, length, tile_offsets, tile_bytecounts, etc.
      - we store the raw IFD entries so we can rewrite them properly later.
      - we store in-memory arrays for tile offsets/bytecounts if found.
    """
    def __init__(self):
        self.entries = []             # list[IFDEntry]
        self.subfile_type = 0
        self.width = 0
        self.length = 0
        self.bits_per_sample = []
        self.compression = 1
        self.photometric = 1
        self.samples_per_pixel = 1
        self.planar_config = 1
        self.tile_width = 0
        self.tile_length = 0
        self.tile_offsets = []
        self.tile_byte_counts = []
        # Additional fields can be added as needed.

    def __repr__(self):
        return (f"<TiffIFD w={self.width} h={self.length} "
                f"tileCount={len(self.tile_offsets)}>"
               )

def read_ifds(f, byte_order):
    """
    Reads all IFDs from an open TIFF file handle, returning a list of TiffIFD objects.
    This function only handles 'classic' (32-bit) TIFF structure with 42 magic.
    """
    # read 2 bytes = byte_order, 2 bytes = magic, 4 bytes = offset to first IFD
    if byte_order == TIFF_BIGENDIAN:
        endian_str = '>'
    else:
        endian_str = '<'

    magic = struct.unpack(endian_str + 'H', f.read(2))[0]
    if magic != TIFF_MAGIC:
        raise ValueError("Not a classic TIFF or unexpected magic number.")

    first_ifd_offset = struct.unpack(endian_str + 'I', f.read(4))[0]
    if first_ifd_offset == 0:
        # No IFD at all?
        return []

    ifds = []
    offset = first_ifd_offset
    while offset != 0:
        f.seek(offset, 0)
        # read number of IFD entries (2 bytes)
        num_entries = struct.unpack(endian_str + 'H', f.read(2))[0]

        # read each entry (12 bytes)
        ifd = TiffIFD()
        for _ in range(num_entries):
            raw = f.read(12)
            tag, ttype, count, value_offset = struct.unpack(endian_str + 'HHII', raw)
            e = IFDEntry(tag, ttype, count, value_offset)
            ifd.entries.append(e)

        # read next IFD offset (4 bytes)
        next_ifd_offset = struct.unpack(endian_str + 'I', f.read(4))[0]
        offset = next_ifd_offset

        # interpret minimal tags we care about:
        parse_tiff_ifd_entries(f, byte_order, ifd, endian_str)

        ifds.append(ifd)

    return ifds

def parse_tiff_ifd_entries(f, byte_order, ifd, endian_str):
    """
    For each relevant entry in the IFD, read the data if needed and populate the TiffIFD object.
    We only care about a handful of tags: tile offsets/bytecounts, width/length, etc.
    """
    # size for each TIFF data type (we only handle a few):
    type_sizes = {
       TYPE_BYTE: 1,
       TYPE_SHORT: 2,
       TYPE_LONG: 4,
    }

    for e in ifd.entries:
        if e.type not in type_sizes:
            # skip unknown or extended types
            continue

        size_per_item = type_sizes[e.type]
        total_size = e.count * size_per_item

        # If total_size <= 4, the data is "inlined" in e.value_offset.
        # Otherwise, we need to go read from offset e.value_offset in the file.
        inline_data = (total_size <= 4)

        # Move file pointer and read the actual data
        if inline_data:
            # The 4 bytes of e.value_offset directly contain the data
            data_offset = None  # no separate offset
            raw_bytes = None

            # We reconstruct the "4 bytes" from e.value_offset:
            # Because e.value_offset is a 32-bit integer, we can pack/unpack it
            packed = struct.pack(endian_str + 'I', e.value_offset)
            raw_bytes = packed[:total_size]  # just the relevant portion
        else:
            data_offset = e.value_offset
            # store current pos
            cur_pos = f.tell()
            f.seek(data_offset, 0)
            raw_bytes = f.read(total_size)
            f.seek(cur_pos, 0)  # restore

        # Now decode raw_bytes according to e.type:
        if e.type == TYPE_BYTE:
            # single or multiple bytes
            vals = list(raw_bytes)
        elif e.type == TYPE_SHORT:
            # read as e.count of 16-bit values
            vals = []
            for i in range(e.count):
                val = struct.unpack(endian_str + 'H', raw_bytes[i*2:(i+1)*2])[0]
                vals.append(val)
        elif e.type == TYPE_LONG:
            # read as e.count of 32-bit values
            vals = []
            for i in range(e.count):
                val = struct.unpack(endian_str + 'I', raw_bytes[i*4:(i+1)*4])[0]
                vals.append(val)
        else:
            vals = []  # skip anything else

        # Store them appropriately in the ifd object
        if e.tag == SUBFILE_TYPE_TAG and len(vals) == 1:
            ifd.subfile_type = vals[0]
        elif e.tag == IMAGE_WIDTH_TAG and len(vals) == 1:
            ifd.width = vals[0]
        elif e.tag == IMAGE_LENGTH_TAG and len(vals) == 1:
            ifd.length = vals[0]
        elif e.tag == BITS_PER_SAMPLE_TAG:
            ifd.bits_per_sample = vals
        elif e.tag == COMPRESSION_TAG and len(vals) == 1:
            ifd.compression = vals[0]
        elif e.tag == PHOTOMETRIC_TAG and len(vals) == 1:
            ifd.photometric = vals[0]
        elif e.tag == SAMPLES_PER_PIXEL_TAG and len(vals) == 1:
            ifd.samples_per_pixel = vals[0]
        elif e.tag == PLANAR_CONFIG_TAG and len(vals) == 1:
            ifd.planar_config = vals[0]
        elif e.tag == TILE_WIDTH_TAG and len(vals) == 1:
            ifd.tile_width = vals[0]
        elif e.tag == TILE_LENGTH_TAG and len(vals) == 1:
            ifd.tile_length = vals[0]
        elif e.tag == STRIP_OFFSETS_TAG:
            ifd.tile_offsets = vals
        elif e.tag == STRIP_BYTE_COUNTS_TAG:
            ifd.tile_byte_counts = vals
        # else: you can handle more tags if needed

def sanity_check_ifd(ifd):
    """
    Ensure the IFD is tiled (has tile offsets), 
    that tile_offsets and tile_byte_counts match in length, etc.
    Raise an error if inconsistent or not a valid tiled TIFF.
    """
    if len(ifd.tile_offsets) == 0 or len(ifd.tile_byte_counts) == 0:
        raise ValueError("No tile offsets/byte counts found—this TIFF might be stripped or invalid.")
    if len(ifd.tile_offsets) != len(ifd.tile_byte_counts):
        raise ValueError("Mismatch between number of tile offsets and tile byte counts.")
    if ifd.width == 0 or ifd.length == 0:
        raise ValueError("Width or Height is zero—invalid TIFF?")
    if ifd.tile_width == 0 or ifd.tile_length == 0:
        raise ValueError("TileWidth/TileLength are zero—invalid or non-tiled TIFF.")

def read_tiff(infile):
    """
    Reads a classic TIFF from 'infile', returns a list of TiffIFD objects
    plus metadata about endianness. Only supports single IFD or multiple, 
    but does no multi-resolution logic here.
    """
    with open(infile, 'rb') as f:
        # read the byte order (2 bytes)
        byte_order = f.read(2)
        if byte_order not in (TIFF_BIGENDIAN, TIFF_LITTLEENDIAN):
            raise ValueError("Invalid TIFF byte order: not 'MM'/'II'?")

        # Now parse IFDs
        ifds = read_ifds(f, byte_order)
        return byte_order, ifds

# infile, outfile, byte_order, ifds = input_tiff, output_tiff, byte_order, ifds
def write_cog(infile, outfile, byte_order, ifds):
    """
    Writes out a new TIFF that is "COG-like":
      - We place the 'ghost' area first, then all IFDs, then tile data.
      - Recompute tile offsets to place them after the IFD block.
      - For simplicity, we only handle one IFD (the first) here, though the code
        can be extended to multiple ifds or subIFDs/overviews.
    """
    # For demonstration, we only rewrite the FIRST IFD in a "COG-like" manner.
    # If you want multiple IFDs (overviews) you'd repeat the logic for each,
    # linking them via nextIFD pointer. Similarly for masks, etc.

    primary_ifd = ifds[0]

    # Basic sanity check
    sanity_check_ifd(primary_ifd)

    # Compute total number of IFD entries we will rewrite
    # (just the ones relevant to basic image reading).
    # Real code would handle more tags (bitsperSample, etc.).
    # We'll definitely store:
    #   SubfileType (254) only if != 0,
    #   ImageWidth(256),
    #   ImageLength(257),
    #   BitsPerSample(258) if present,
    #   Compression(259),
    #   Photometric(262),
    #   SamplesPerPixel(277),
    #   PlanarConfig(284),
    #   TileWidth(322),
    #   TileLength(323),
    #   TileOffsets(324),
    #   TileByteCounts(325).
    # In a real code, you'd match exactly the logic from the Go source re: required tags.

    tags_to_write = []
    def maybe(tag):  # check if the tag was found in the IFD
        return any(e.tag == tag for e in primary_ifd.entries)

    # Build an in-memory set of the tags we want to write:
    candidate_tags = [
        SUBFILE_TYPE_TAG,
        IMAGE_WIDTH_TAG, IMAGE_LENGTH_TAG,
        BITS_PER_SAMPLE_TAG,
        COMPRESSION_TAG, PHOTOMETRIC_TAG,
        SAMPLES_PER_PIXEL_TAG, PLANAR_CONFIG_TAG,
        TILE_WIDTH_TAG, TILE_LENGTH_TAG,
        STRIP_OFFSETS_TAG, STRIP_BYTE_COUNTS_TAG
    ]
    # Filter only if they make sense
    for t in candidate_tags:
        if t == SUBFILE_TYPE_TAG and primary_ifd.subfile_type == 0:
            continue
        if t == BITS_PER_SAMPLE_TAG and not primary_ifd.bits_per_sample:
            continue
        # We skip if the original had no offsets, which shouldn't happen if it's tiled
        # ... or skip if we're definitely rewriting them anyway (we do handle them)
        tags_to_write.append(t)

    # Let's produce the in-memory “new offset” for tile data.
    # We'll place the tile data after:
    #   - 8 bytes of TIFF header
    #   - length of the GHOST area
    #   - the IFD block itself
    #
    # We'll store tile data in the order they appear in tile_offsets,
    # each tile preceded by 4 bytes (size) and followed by 4 bytes repeated.

    # Roughly compute the size of the IFD block:
    #   - 2 bytes for the num_entries
    #   - 12 bytes per entry
    #   - 4 bytes for nextIFD offset
    ifd_entry_count = len(tags_to_write)
    ifd_block_size = 2 + 12 * ifd_entry_count + 4

    # The new file layout:
    #  offset 0..1: Byte order
    #  offset 2..3: magic (42)
    #  offset 4..7: offset to first IFD (just after GHOST)
    #  offset 8.. : GHOST data
    #  ...
    #  then the IFD
    #  then tile data
    #
    # We'll write offsets in little-endian or big-endian according to the input.

    if byte_order == TIFF_BIGENDIAN:
        endian_str = '>'
    else:
        endian_str = '<'

    # Build the output in binary form:
    out = open(outfile, 'wb')

    # 1) Write the TIFF header
    out.write(byte_order)  # 'II' or 'MM'
    out.write(struct.pack(endian_str + 'H', TIFF_MAGIC))  # 42
    offset_to_ifd = 8 + len(GHOST)
    out.write(struct.pack(endian_str + 'I', offset_to_ifd))

    # 2) Write the GHOST area
    out.write(GHOST)

    # 3) Write the IFD itself (placeholder), but let's also figure out
    #    how big the IFD is to find the start of tile data.
    #    We'll write it with placeholder offsets for tile offsets array, etc.,
    #    then come back and fix them. For simplicity, let's just build
    #    the entire IFD block in memory, then write it once.

    ifd_start = out.tell()
    # Write the number of entries
    out.write(struct.pack(endian_str + 'H', ifd_entry_count))
    # We'll store the offset for rewriting each entry.
    entry_positions = []

    for tag in tags_to_write:
        entry_positions.append(out.tell())
        out.write(b'\x00' * 12)  # placeholder 12 bytes
    # nextIFD offset
    out.write(struct.pack(endian_str + 'I', 0))  # no more IFDs in this minimal example

    # keep track of the current "overflow" offset if we had arrays bigger than 4 bytes
    # but let's do a naive approach: store bitsPerSample inlined if 1 or 2, otherwise do minimal.
    # For tile offsets/bytecounts, we definitely need an offset array in the "post-IFD" area.

    # Let's define a small "overflow" buffer in memory, then we know where it starts:
    overflow_data = bytearray()
    overflow_pos_in_ifd = 0  # not used in this minimal example

    # 4) We'll compute where tile data will start: right after the entire IFD block.
    tile_data_start = ifd_start + ifd_block_size
    # We'll place the tile offsets array just before the tile data (for simplicity).
    # The size of the tile offsets array is 4 * len(tile_offsets).
    # Similarly for tile bytecounts array.
    num_tiles = len(primary_ifd.tile_offsets)
    # We'll have two arrays: tileOffsets[] and tileByteCounts[] each 4*num_tiles in size.

    # So the "pointer area" (for offsets/bytecounts) is 8 * num_tiles in total
    pointer_area_size = 8 * num_tiles
    # We'll put the tileOffsets array, then the tileByteCounts array in that region.

    tile_offsets_array_offset = tile_data_start
    tile_bytecounts_array_offset = tile_offsets_array_offset + 4 * num_tiles

    # Then the actual tile data will start at:
    actual_tile_data_start = tile_bytecounts_array_offset + 4 * num_tiles

    # We'll rewrite the tile offsets so that each tile is placed sequentially from actual_tile_data_start onward.

    new_tile_offsets = []
    # We'll track the next free position for tile data
    next_tile_data_pos = actual_tile_data_start

    # 5) Rebuild tile data in the new file
    # We'll gather them, rewriting each tile:
    #   4 bytes: tile size
    #   tile bytes
    #   4 bytes: repeat last 4 bytes
    #
    # But let's hold off writing tile data. We first finish the IFD area, then come back.

    # For convenience, read the tile data from the input
    # We'll create a structure to hold (offset, size, data).
    tile_blobs = []
    with open(infile, 'rb') as fin:
        for i, off in enumerate(primary_ifd.tile_offsets):
            size = primary_ifd.tile_byte_counts[i]
            if size == 0:
                # skip empty tile
                tile_blobs.append(None)
                new_tile_offsets.append(0)
                continue
            fin.seek(off, 0)
            tile_bytes = fin.read(size)
            tile_blobs.append(tile_bytes)

    # 6) Fill in the IFD entries.
    def write_ifd_entry(pos, tag, ttype, count, value):
        """
        Overwrite the placeholder with the real 12-byte IFD entry:
          [0..2): tag
          [2..4): type
          [4..8): count
          [8..12): value or offset
        `value` is a 4-byte integer or smaller data inlined. 
        """
        # We do everything in memory, then out.seek and write once.
        cur = out.tell()
        out.seek(pos, 0)
        out.write(struct.pack(endian_str + 'H', tag))
        out.write(struct.pack(endian_str + 'H', ttype))
        out.write(struct.pack(endian_str + 'I', count))
        out.write(struct.pack(endian_str + 'I', value))
        out.seek(cur, 0)

    # We build a dictionary from tag->entry_pos so we can fill them in easily:
    tag2pos = dict(zip(tags_to_write, entry_positions))

    # Utility to pick the "first" value from a list if needed
    def first_or_zero(lst):
        return lst[0] if lst else 0

    # SUBFILE_TYPE (254) if not zero:
    if SUBFILE_TYPE_TAG in tag2pos:
        write_ifd_entry(tag2pos[SUBFILE_TYPE_TAG],
                        SUBFILE_TYPE_TAG, TYPE_LONG, 1,
                        primary_ifd.subfile_type)

    # IMAGE_WIDTH (256), assume 1 value
    if IMAGE_WIDTH_TAG in tag2pos:
        write_ifd_entry(tag2pos[IMAGE_WIDTH_TAG],
                        IMAGE_WIDTH_TAG, TYPE_LONG, 1, primary_ifd.width)

    # IMAGE_LENGTH (257)
    if IMAGE_LENGTH_TAG in tag2pos:
        write_ifd_entry(tag2pos[IMAGE_LENGTH_TAG],
                        IMAGE_LENGTH_TAG, TYPE_LONG, 1, primary_ifd.length)

    # BITS_PER_SAMPLE (258) - if more than 2 samples, we can't inline easily in 4 bytes;
    # let's just inline if there's only 1 sample. Otherwise, do the naive approach:
    if BITS_PER_SAMPLE_TAG in tag2pos:
        bps = primary_ifd.bits_per_sample
        if len(bps) == 1:
            write_ifd_entry(tag2pos[BITS_PER_SAMPLE_TAG],
                            BITS_PER_SAMPLE_TAG, TYPE_SHORT, 1, bps[0])
        else:
            # For simplicity, let's store only the first.
            # Real code would store them properly in an array in the overflow area.
            write_ifd_entry(tag2pos[BITS_PER_SAMPLE_TAG],
                            BITS_PER_SAMPLE_TAG, TYPE_SHORT, 1, bps[0])

    # COMPRESSION (259)
    if COMPRESSION_TAG in tag2pos:
        write_ifd_entry(tag2pos[COMPRESSION_TAG],
                        COMPRESSION_TAG, TYPE_SHORT, 1, primary_ifd.compression)

    # PHOTOMETRIC (262)
    if PHOTOMETRIC_TAG in tag2pos:
        write_ifd_entry(tag2pos[PHOTOMETRIC_TAG],
                        PHOTOMETRIC_TAG, TYPE_SHORT, 1, primary_ifd.photometric)

    # SAMPLES_PER_PIXEL (277)
    if SAMPLES_PER_PIXEL_TAG in tag2pos:
        write_ifd_entry(tag2pos[SAMPLES_PER_PIXEL_TAG],
                        SAMPLES_PER_PIXEL_TAG, TYPE_SHORT, 1, primary_ifd.samples_per_pixel)

    # PLANAR_CONFIG (284)
    if PLANAR_CONFIG_TAG in tag2pos:
        write_ifd_entry(tag2pos[PLANAR_CONFIG_TAG],
                        PLANAR_CONFIG_TAG, TYPE_SHORT, 1, primary_ifd.planar_config)

    # TILE_WIDTH (322)
    if TILE_WIDTH_TAG in tag2pos:
        write_ifd_entry(tag2pos[TILE_WIDTH_TAG],
                        TILE_WIDTH_TAG, TYPE_LONG, 1, primary_ifd.tile_width)

    # TILE_LENGTH (323)
    if TILE_LENGTH_TAG in tag2pos:
        write_ifd_entry(tag2pos[TILE_LENGTH_TAG],
                        TILE_LENGTH_TAG, TYPE_LONG, 1, primary_ifd.tile_length)

    # STRIP_OFFSETS (324) => tileOffsets. We store an offset to the array in the file.
    if STRIP_OFFSETS_TAG in tag2pos:
        write_ifd_entry(tag2pos[STRIP_OFFSETS_TAG],
                        STRIP_OFFSETS_TAG, TYPE_LONG, num_tiles,
                        tile_offsets_array_offset)

    # STRIP_BYTE_COUNTS (325) => tileByteCounts. Similar approach.
    if STRIP_BYTE_COUNTS_TAG in tag2pos:
        write_ifd_entry(tag2pos[STRIP_BYTE_COUNTS_TAG],
                        STRIP_BYTE_COUNTS_TAG, TYPE_LONG, num_tiles,
                        tile_bytecounts_array_offset)

    # 7) Now write the "pointer area" (tileOffsets[], tileByteCounts[])
    #    at tile_data_start. We'll do it before writing actual tile data.

    # We'll move to tile_data_start:
    out.seek(tile_data_start, 0)

    # We'll compute new tile offsets & tile bytecounts
    new_tile_bytecounts = []
    for i, tile_data in enumerate(tile_blobs):
        if tile_data is None:
            new_tile_offsets.append(0)
            new_tile_bytecounts.append(0)
        else:
            new_tile_offsets.append(next_tile_data_pos)
            # the "wrapped" size is size + 8 (leading 4 bytes + trailing 4 bytes).
            tile_size = len(tile_data)
            # We store the "compressed tile" size ignoring the 8 bytes overhead.
            # That overhead is a COG block leader/trailer, not counted in the TIFF's tileByteCounts.
            new_tile_bytecounts.append(tile_size)
            next_tile_data_pos += tile_size + 8  # 8 extra bytes for block leader/trailer

    # Write the new_tile_offsets as 32-bit each:
    for offv in new_tile_offsets:
        out.write(struct.pack(endian_str + 'I', offv))
    # Then tile bytecounts:
    for bcv in new_tile_bytecounts:
        out.write(struct.pack(endian_str + 'I', bcv))

    # 8) Finally, write actual tile data blocks after the pointer area:
    for i, tile_data in enumerate(tile_blobs):
        if tile_data is None or len(tile_data) == 0:
            continue
        # The location we decided above:
        out.seek(new_tile_offsets[i], 0)
        size_bytes = struct.pack('<I', len(tile_data))  # always little-end?
        # For strict correctness, you might do `endian_str + 'I'`, but typically
        # the block leader is a "private" 4 bytes for the COG layout, not standard TIFF data.
        out.write(size_bytes)
        out.write(tile_data)
        # repeat last 4 bytes
        out.write(tile_data[-4:] if len(tile_data) >= 4 else tile_data)

    out.close()

def cogwriter(input_tiff, output_tiff):
    """
    A minimal "from scratch" function that:
      1) Reads the input TIFF in classic 32-bit format,
      2) Parses the first IFD (tiles),
      3) Writes a new output "COG-like" TIFF with tile data placed behind the IFD.
         Offsets are updated accordingly, plus a 'ghost' area is placed after the header.
    """
    # parse input
    byte_order, ifds = read_tiff(infile=input_tiff)
    if not ifds:
        raise ValueError("No IFDs found in input TIFF!")

    # rewrite
    write_cog(input_tiff, output_tiff, byte_order, ifds)

    return output_tiff

# Example usage (directly call cogwriter).
# You can rename or comment this out if you just want the library portion.
if __name__ == "__main__":
    input_tiff = "/home/cesar/Desktop/DEMO/BSQ_tiled.tiff"
    output_tiff = "/home/cesar/Desktop/DEMO/BSQ_tiled2.tiff"
    cogwriter(input_tiff, output_tiff)
    print(f"COG written to: {output_tiff}")





