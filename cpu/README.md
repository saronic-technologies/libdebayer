# Debayer on CPU

## Setup

On Linux:

```bash
sudo apt update
sudo apt install -y build-essential cmake git libopencv-dev
```

On Mac:

```bash
brew install opencv llvm libomp
```

## Build

```
mkdir build
cd build
cmake ..
make -j
```
