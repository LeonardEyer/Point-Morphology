# Point Morphology

This repository aims to implement Point Morphology, a way to perfom morphological operations on pointclouds.


## Building

Was built and tested with

- cmake 4.3.0
- applce clang 21.0.0

```bash
cmake -B build -S . -GNinja -DCMAKE_BUILD_TYPE=Release
```

## Usage

For running in interactive gui mode just supply a pointcloud
```bash
./point-morphology -[noff/ply] [path-to-file]
```
