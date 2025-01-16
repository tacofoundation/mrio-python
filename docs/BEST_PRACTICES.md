# Best Practices

This section provides guidance on when and when not to use the Multidimensional GeoTIFF and Temporal GeoTIFF formats.

## When to use it?

#### Multidimensional or Temporal GeoTIFF

Ideal for machine learning [mini-cubes](https://doi.org/10.1017/eds.2024.22) workflows, especially when each sample should be retrieved in a single operation. It also excels at sharing data with non-specialized users, offering seamless access and compatibility with commonly used geospatial tools.

#### NetCDF, HDF5, or Zarr

Ideal for complex data analysis workflows, these formats provide superior flexibility, supporting nested groups and advanced chunking strategies. They are ideal for storing large datacubes with detailed metadata.


#### Safetensor

The best and safe option for storing and sharing tensor data. The API is available in Rust with a Python wrapper, but as the format is so simple, it can be easily implemented in other languages. Most of the core feature are possible because it does not implement a store compression schemes. As no compression is used, the data can ocupy more space than other formats, specially in a remote sensing context where auto-correlation is very common.


## Analysing Performance

We evaluate the **READ** performance of the Multidimensional GeoTIFF format by comparing it to NetCDF, HDF5, Zarr, Sharded Zarr, and Zipped Zarr. We define four different use cases tipical in remote sensing applications:

    - Spatial Query: Read a specific spatial region and specific spectral band from the datacube.
    - Spectral-Spatial Query: Read a six spectral bands from a specific spatial region.
    - Spectral-Spatial-Temporal Query: Read a specific spatial region and four spectral bands from a specific time period.
    - Simulated-Spectral-Spatial-Temporal Query: Read a specific spatial region and eight spectral bands from a specific time period for two different data simulations.


We set up the same chunk strategy for HDF5 and Zarr format, chunks are no smaller than 1Mb. As the Multidimensional GeoTIFF format does not support multi-dimensional chunking, we set up the 





on Local Disk


two possible scenarios: a **spatial chunking strategy** and an **spatial-temporal chunking strategy**. We consider a 5D datacube with the following dimensions: `simulation`, `time`, `band`, `x`, and `y`. 

The datacube has the following shape: 
    - small (10Mb): 3 x 5 x 8 x 512 x 512
    - medium (100Mb): 3 x 200 x 8 x 512 x 512
    - large (1Gb): 3 x 200 x 8 x 2048 x 2048


### Generate a Toy Example


### Simple Chunking Strategy



### Advanced Chunking Strategy


## Analysing Performance on HTTP 1.0 Server


