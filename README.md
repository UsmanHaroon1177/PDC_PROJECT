# Parallel Multi-Objective Shortest Path in Dynamic Networks

*Authors*: 22i-0928, 22i-1177, 22i-0764

## Overview

This repository provides a high-performance hybrid *MPI+OpenMP* implementation of dynamic single-objective and multi-objective shortest-path updates on large graphs. It builds on the SOSP\_Update and MOSP\_heuristic algorithms from Khanda et al. (SC-W 2023) and integrates METIS-based partitioning for distributed scalability.

### Key Components

* *Distributed Partitioning & Initial SOSP*

  * Graph partitioning via METIS
  * Hybrid MPI + OpenMP Bellman–Ford for initial Single-Source Shortest Path (SOSP)

* *Shared-Memory SOSP\_Update*

  * Two-phase grouping relaxation for edge insertions
  * Minimal propagation scoped only to affected vertices

* *MOSP Heuristic*

  * Pareto-path heuristic combining per-objective SOSP trees into a compact ensemble graph
  * Single-run Dijkstra to extract one representative multi-objective path

## Features

* Scales to millions of vertices and edges across MPI ranks and OpenMP threads
* Negligible update cost when edge insertions do not change distances
* Flexible configuration: number of MPI ranks, OpenMP threads, METIS options

## Prerequisites

* C++17 compiler (e.g., g++, mpicxx)
* MPI library (e.g., MPICH, OpenMPI)
* OpenMP support
* METIS v5

## Building

bash
mkdir build && cd build
cmake ..              # or configure paths to MPI and METIS
make


## Running

bash
mpirun -np <num_ranks> ./dynamic_mosp path/to/graph.mtx --allow-noncontig -t <threads_per_rank>


### Example

bash
mpirun -np 6 ./dynamic_mosp ../road-roadNet-CA.mtx --allow-noncontig -t 16


Outputs:

* Sample global distances printed to console
* Full distance vectors written to distances.csv and distances_after_update.csv
* Phase timings, edgecut, and update statistics summarized at end

## Profiling and Hotspot Analysis

* *gprof*: build with -pg, run to collect gmon.out, then gprof ./dynamic_mosp gmon.out > analysis.txt.
* *mpiP / Intel Trace Analyzer*: analyze MPI_Alltoallv volumes and imbalance.
* *VTune / gprof*: identify OpenMP critical sections and vector/iterator overhead.

## Performance Insights

* *Initial SOSP (Bellman–Ford)* dominates runtime (\~50%). Communication-bound by all-to-all updates.
* *SOSP\_Update* is negligible when no distances change; costs grow with affected subgraph size.
* *MOSP Heuristic* runs in milliseconds on the root; ensemble graph is orders of magnitude smaller than full graph.



## Contributing

1. Fork the repository
2. Create a topic branch (git checkout -b feature/XYZ)
3. Commit your changes (git commit -am 'Add new feature')
4. Push to the branch (git push origin feature/XYZ)
5. Open a Pull Request

Please adhere to existing code style and include new tests where applicable.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
