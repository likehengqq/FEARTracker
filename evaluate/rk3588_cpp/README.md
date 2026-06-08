# FEAR RK3588 C++ inference demo

This directory contains a C++ RKNN inference demo for RK3588 boards. It uses:

- Rockchip RKNN C runtime (`rknn_api.h`, `librknnrt.so`)
- OpenCV for video I/O, resize, padding, color conversion, and drawing

The demo expects the two `.rknn` files exported by `evaluate/rk3588_export.py`:

- `fear_template_encoder.rknn`
- `fear_track.rknn`

## Build on RK3588

Install OpenCV development files and Rockchip RKNN runtime first. If RKNN headers and libraries are not in a system path, point CMake to the runtime package with `RKNN_API_PATH`.

```shell
cd evaluate/rk3588_cpp
cmake -S . -B build -DRKNN_API_PATH=/path/to/rknn/runtime
cmake --build build -j
```

If `rknn_api.h` and `librknnrt.so` are already installed under `/usr/include` and `/usr/lib`, `-DRKNN_API_PATH=...` can be omitted.

## Run

```shell
./build/fear_rk3588_demo \
  --template_model ../../outputs/rk3588/fear_template_encoder.rknn \
  --track_model ../../outputs/rk3588/fear_track.rknn \
  --video ../../assets/test.mp4 \
  --output ../../outputs/rk3588_cpp/test.mp4 \
  --bbox 163,53,45,174 \
  --core auto
```

Options:

- `--bbox x,y,w,h`: initial object box in the first frame
- `--core auto|0|1|2|all`: RK3588 NPU core mask

OpenCV reads video frames as BGR. The demo converts each frame to RGB before model preprocessing, matching the Python tracker normalization.
