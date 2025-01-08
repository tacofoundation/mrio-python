The terms “MUST”, “MUST NOT”, “REQUIRED”, “SHALL”, “SHALL NOT”, “SHOULD”, “SHOULD NOT”, “RECOMMENDED”, “MAY”, and “OPTIONAL” in this document follow the definitions from [RFC 2119](https://tools.ietf.org/html/rfc2119).

# The Multidimensional GeoTIFF specification

## Overview

Most of the existing geospatial data formats for multidimensional arrays, such as NetCDF, HDF5, or Zarr, are not natively supported by GIS software or geospatial libraries. Besides, they do not fully integrate with the GDAL library, which is the de facto standard for reading and writing raster data in the geospatial community.

The Multidimensional GeoTIFF (mGeoTIFF) specification extends the traditional GeoTIFF format to support N-dimensional arrays. It maintains the simplicity and compatibility of GeoTIFF, offering fast access and the ability to be opened by any GIS software or library that supports the GeoTIFF format.

## Format

This is the version `0.1.0` of the mGeoTIFF specification. The main difference between a [traditional GeoTIFF](https://docs.ogc.org/is/19-008r4/19-008r4.html) and an mGeoTIFF is the addition of a global 
TAGS named as **`MD_METADATA`** that contains the metadata of the multidimensional array. The `MD_METADATA` tag is a encoded JSON string that contains the following keys:

Field  | Type | Required | Details |
|---|---|---|---|
| md:pattern | string | Yes | A string defining the strategy to reshape the data into a 3D array (band, x, y). It is based on the Einstein-Inspired Notation for OPerationS, einops. The pattern is a space-separated list of dimension names, followed by an arrow `->`, and the new order of the dimensions. For example, `time band lat lon -> (time band) lat lon` rearranges the dimensions from `(time, band, lat, lon)` to `timexband lat lon`, where `timexband` is a new number of channels. Refer to the [einops paper](https://openreview.net/pdf?id=oapKSVM2bcj) for more details. |
| md:coordinates | dictionary | Yes | A dictionary defining the coordinates to be combined with the pattern. The values **MUST** be lists of data types compliant with the [JSON standard](https://www.json.org/json-en.html). |
| md:attributes | dictionary | No | A dictionary of additional metadata attributes to include in the file. It **MUST** complies with the [JSON standard](https://www.json.org/json-en.html). |
| md:dimensions | list | No | A list of dimension names, where the order **MUST** align with the order specified in `md:pattern` before the arrow `->`. |
| md:coordinates_len | dictionary | No | A dictionary defining the length of each dimension. The values **MUST** be integers. |

## Example

The following is an example of the `MD_METADATA` tag in a mGeoTIFF file:

```json
{
  "md:pattern": "time band lat lon -> (time band) lat lon",
  "md:coordinates": {
    "time": ["2021-01-01", "2021-01-02", "2021-01-03"],
    "band": ["B01", "B02", "B03"]
  },
  "md:dimensions": ["time", "band", "lat", "lon"],
  "md:attributes": {
    "title": "Multidimensional GeoTIFF Example",
    "description": "This is a toy example of a Multidimensional GeoTIFF file."
  },
    "md:coordinates_len": {
        "time": 3,
        "band": 3,
        "lat": 100,
        "lon": 100
    }
}
```

# The Temporal GeoTIFF specification

## Overview

Temporal mini-datacubes are becoming increasingly popular in the Earth Observation community. They are used to store time series of satellite images, facilitating the analysis of land cover changes, crop monitoring, and other applications. However, there is no standard format for storing and sharing these datacubes. Current formats lack an explicit convention for defining the dimensions of a temporal mini-datacube.

The temporal GeoTIFF (tGeoTIFF) specification refines the mGeoTIFF format by enforcing a more stringent convention for defining its dimensions.

## Format

This is the version `0.1.0` of the tGeoTIFF specification. The main difference between a mGeoTIFF and a tGeoTIFF is that it **MUST** include four dimensions in the following order, with the specified naming convention: `(time, band, x, y)`. Additionally, certain attributes **MUST** be included in the `md:attributes` field of the **`MD_METADATA`** tag.

| Attribute | Type | Required | Details |
|---|---|---|---|
| md:time_start | Long | Yes | The nominal start time of acquisition. It **MUST** be expressed as a Unix timestamp in seconds. |
| md:id | String | Yes | A unique identifier for each time step. Therefore, it **MUST** have the same length as the `md:time_start` and the `time` dimension. |
| md:time_end | Long | No | The nominal end time of the acquisition or composite period. It **MUST** be expressed as a Unix timestamp in seconds. It **MUST** have the same length as the `md:time_start` and the `time` dimension. |

## Example

The following is an example of the `MD_METADATA` tag in a tGeoTIFF file:

```json
{
  "md:pattern": "time band lat lon -> (time band) lat lon",
  "md:coordinates": {
    "time": ["2021-01-01", "2021-01-02", "2021-01-03"],
    "band": ["B01", "B02", "B03"]
  },
  "md:dimensions": ["time", "band", "lat", "lon"],
  "md:attributes": {
    "title": "Temporal GeoTIFF Example",
    "description": "This is a toy example of a Temporal GeoTIFF file.",
    "md:id": [
        "S2A_MSIL2A_20210101T101021_N0214_R022_T33UYP_20210101T103000",
        "S2A_MSIL2A_20210102T101021_N0214_R022_T33UYP_20210102T103000",
        "S2A_MSIL2A_20210103T101021_N0214_R022_T33UYP_20210103T103000"
    ],
    "md:time_start": [
        1609481400,
        1609567800,
        1609654200
    ],
    "md:time_end": [
        1609567800,
        1609654200,
        1609740600
    ]
  },
    "md:coordinates_len": {
        "time": 3,
        "band": 3,
        "lat": 100,
        "lon": 100
    }
}
```