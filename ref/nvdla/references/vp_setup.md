# NVDLA Virtual Platform Setup

> **Important context**: Read `VP_EXPLAINED.md` first for an analysis of how the VP works, why building it on RK3588 is not useful, and alternative approaches to using NVDLA knowledge for conv.py debugging.

## Purpose

The NVDLA VP (Virtual Platform) simulates the full NVDLA hardware stack on an x86 machine. It uses:
- **SystemC 2.3.0a** — C++ simulation framework
- **QEMU** — ARMv8 processor emulation (wrapped as TLM-2.0 module)
- **GreenSocs QBox** — QEMU/SystemC cosimulation infrastructure
- **NVDLA C-model** — Cycle-accurate SystemC model of NVDLA hardware

The VP allows running the full NVDLA software stack (compiler + runtime) without real hardware.

## Build Steps (x86 Only)

### 1. Prerequisites

```bash
sudo apt-get install g++ cmake libboost-dev python-dev libglib2.0-dev \
    libpixman-1-dev liblua5.2-dev swig libcap-dev libattr1-dev libncurses5-dev
```

### 2. SystemC 2.3.0a

```bash
wget -O systemc-2.3.0a.tar.gz http://www.accellera.org/images/downloads/standards/systemc/systemc-2.3.0a.tar.gz
tar -xzvf systemc-2.3.0a.tar.gz
cd systemc-2.3.0a
sudo mkdir -p /usr/local/systemc-2.3.0/
mkdir objdir && cd objdir
../configure --prefix=/usr/local/systemc-2.3.0
make -j$(nproc)
sudo make install
```

### 3. Perl YAML + IO-Tee

```bash
wget -O YAML-1.24.tar.gz http://search.cpan.org/CPAN/authors/id/T/TI/TINITA/YAML-1.24.tar.gz
tar -xzvf YAML-1.24.tar.gz && cd YAML-1.24
perl Makefile.PL && make && sudo make install
cd ..
wget -O IO-Tee-0.65.tar.gz http://search.cpan.org/CPAN/authors/id/N/NE/NEILB/IO-Tee-0.65.tar.gz
tar -xzvf IO-Tee-0.65.tar.gz && cd IO-Tee-0.65
perl Makefile.PL && make && sudo make install
```

### 4. NVDLA HW (CMOD)

```bash
git clone https://github.com/nvdla/hw.git
cd hw
make   # will prompt for tool paths
# Set java, g++, perl, systemc paths as prompted
# For small config: DEFAULT_PROJ:=nv_small
tools/bin/tmake -build cmod_top
```

### 5. Build VP

```bash
git clone https://github.com/nvdla/vp.git
cd vp
git submodule update --init --recursive
```

**Patch CMakeLists.txt** (for modern gcc):
Remove `-Werror` from:
- `vp/CMakeLists.txt`
- `vp/models/nvdla/CMakeLists.txt`
- `vp/fpga/aws-fpga/fpga_sc_wrapper/CMakeLists.txt`
- `vp/fpga/aws-fpga/cosim_sc_wrapper/CMakeLists.txt`

```bash
cmake -DCMAKE_INSTALL_PREFIX=build \
      -DSYSTEMC_PREFIX=/usr/local/systemc-2.3.0/ \
      -DNVDLA_HW_PREFIX=/path/to/hw \
      -DNVDLA_HW_PROJECT=nv_small
make -j$(nproc)
make install
```

### 6. Obtain Kernel Image

Option A — Prebuilt (from nvdla/sw repo):
```bash
# Use Image from sw/prebuilt/arm64-linux/image/linux/
```

Option B — Build with buildroot:
```bash
wget https://buildroot.org/downloads/buildroot-2017.11.tar.gz
tar -xzvf buildroot-2017.11.tar.gz && cd buildroot-2017.11/
make qemu_aarch64_virt_defconfig
make menuconfig
# Set: Target → AArch64, Toolchain → Linaro AArch64 2017.08, Kernel → 4.13.3
make -j$(nproc)
```

### 7. Run

```bash
# Edit conf/aarch64_nvdla.lua to point to your kernel and rootfs
export SC_SIGNAL_WRITE_CHECK=DISABLE
./build/bin/aarch64_toplevel -c conf/aarch64_nvdla.lua

# Inside QEMU shell:
mount -t 9p -o trans=virtio r /mnt
cd /mnt/tests/hello
./aarch64_hello
```

## Compiling + Running Inference on VP

```bash
# Compile Caffe model → NVDLA loadable
./nvdla_compiler --profile fast-math \
    --configtarget nv_small \
    --cprecision int8 \
    --prototxt resnet50.prototxt \
    --caffemodel resnet50.caffemodel \
    --calibtable resnet50.json

# Run inference
./nv_runtime --loadable fast_math.nvdla \
    --image test.jpg \
    --rawdump --normalize 1
```

## Known Issues

1. **template 'auto_ptr' deprecated**: Remove `-Werror` from CMakeLists.txt
2. **memfd.c compile error**: Patch `vp/libs/qbox/lib/backdoor/memfd.c` for newer glibc
3. **Port forwarding**: Comment out `tcp::6666-:6666` in lua file if port unavailable
4. **Slow simulation**: ~2 hours per inference with nv_full
5. **nv_small instability**: May drop mid-inference; nv_large or nv_full more stable

## References

- [Official VP docs](http://nvdla.org/vp.html)
- [NVDLA VP GitHub](https://github.com/nvdla/vp)
- [NVDLA HW GitHub](https://github.com/nvdla/hw)
- [NVDLA SW GitHub](https://github.com/nvdla/sw)
