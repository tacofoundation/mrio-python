# The Multidimensional GeoTIFF specification

The terms “MUST”, “MUST NOT”, “REQUIRED”, “SHALL”, “SHALL NOT”, “SHOULD”, “SHOULD NOT”, “RECOMMENDED”, “MAY”, and “OPTIONAL” in this document follow the definitions from RFC 2119.


## Overview

The Multidimensional GeoTIFF (GeoTIFF) is a standard for encoding multidimensional arrays in a GeoTIFF file. The GeoTIFF standard is widely used in the geospatial community for encoding raster data. 


## Why Multidimensional GeoTIFF?

Most of the existing geospatial data formats for multidimensional arrays, such as NetCDF, HDF5, or Zarr, are not natively supported by GIS software or geospatial libraries. Besides, they do not integrate well with the GDAL library, which is the de facto standard for reading and writing raster data in the geospatial community.

## Format

This is the version `0.0.1` of the Multidimensional GeoTIFF specification. The specification is subject to change, and we encourage feedback from the community. There are two main components of the Multidimensional GeoTIFF format:

- **Rearrangement Strategy:** A string that defines the order of the dimensions in the data array. The rearrangement strategy is a space-separated list of dimension names, followed by an arrow `->`, and the new order of the dimensions. For example, `time band lat lon -> (time band) lat lon` rearranges the dimensions from `(time, band, lat, lon)` to `(time, band) lat lon`.


- Save rearranged metadata





