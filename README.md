# BLDD research artifact

This package contains the BLDD implementation for the paper's Arc, ArcNext and DirectNeighbors experiments, including UVT and UTV variable orders. Query output reports average time per query in microseconds. Errors are written to standard error. `compress-save` writes the compressed representation and does not print query results.

## Build

Requirements: Windows, Visual Studio 2022 C++ build tools and x64. The bundled modified CUDD 3.0.0 is built automatically from source as part of the solution. It includes the `ID` field in `DdNode`; use the bundled headers and library together because this change affects the node layout. CUDD source files retain their copyright and license notices.

In a Visual Studio developer PowerShell, run:

```powershell
msbuild BLDD.sln /p:Configuration=Release /p:Platform=x64 /m
```

AVX2 support is required by the Release configuration. Open a Visual Studio developer PowerShell if `msbuild` is not on PATH.

## Run the example

Run from the `BLDD` directory so that relative sample paths resolve:

```powershell
cd BLDD
..\x64\Release\BLDD.exe load-query sample
..\x64\Release\BLDD.exe compress-query sample UVT
..\x64\Release\BLDD.exe compress-query sample UTV
```

The bundled graph is a small point-event example, not one of the paper's benchmark datasets. Its measured times are not the paper's results. The UVT example is sorted by source, target, time; the UTV example contains the same records sorted by source, time, target. The first line contains maximum node ID, number of events, and maximum timestamp; subsequent lines contain source, target, timestamp.

## Run another dataset

```text
BLDD.exe <compress-save|compress-query|load-query> <dataset-name> <raw-file> <compressed-directory> [UVT|UTV]
```

Dataset names select time semantics: `yahoo` and `editwiki` use point events; `flickr` uses incremental events; `wikilink`, `cbtpow` (Powerlaw) and `cbtcomm` (Comm.Net) use intervals. Provide the relevant dataset and compressed-directory paths explicitly. Input records must be sorted for the selected variable order; duplicate triples are treated as one event. The header event count includes duplicate records. UVT requires a non-empty input. `compress-save` creates the compressed files, `compress-query` builds in memory and runs the queries, and `load-query` reads compressed files and runs the queries. The raw file is still required for its header when loading. For custom point-event inputs, use the raw file path in place of the dataset name and omit the additional raw-file argument.

The full datasets and external baseline implementations are not included. Default benchmark settings in `Graph.h` use 100,000 queries and five rounds for Arc and DirectNeighbors, and 100 queries and one round for ArcNext. These settings and the generated workloads must be checked against the original experiment records before claiming reproduction of paper figures. The sample does not validate the reported benchmark numbers.

## Dataset provenance

The full experimental datasets are not distributed in this repository. Flickr originates from the [MPI-SWS WWW 2009 dataset](https://socialnetworks.mpi-sws.org/data-www2009.html). Yahoo, Wiki-Edit, Powerlaw, and Comm.Net were provided by the authors of *On Compressing Temporal Graphs* (ICDE 2022); permission to redistribute the supplied files has not been confirmed. Wiki-Links was reconstructed from [English Wikipedia dumps](https://dumps.wikimedia.org/enwiki/). The exact dump version and reconstruction procedure are not included here, so the paper's Wiki-Links dataset cannot currently be reconstructed from this package alone. Access to the other full datasets and the preprocessing needed to reproduce them must also be independently established.

## Regression test

Run from the solution directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File BLDD\tests\run_next_activation_regression.ps1
```

This checks next-activation behavior for a small UVT graph; it is not a full validation of all datasets or query semantics.
