# FEAR RK3588 C++ 推理 demo

本目录提供面向 RK3588 开发板的 C++ RKNN 推理 demo，使用：

- 瑞芯微 RKNN C 运行时（`rknn_api.h`、`librknnrt.so`）
- OpenCV，用于视频读写、缩放、padding、颜色转换与画框

该 demo 需要由 `evaluate/rk3588_export.py` 导出的两个 `.rknn` 文件：

- `fear_template_encoder.rknn`
- `fear_track.rknn`

## 在 RK3588 上编译

请先安装 OpenCV 开发文件与瑞芯微 RKNN 运行时。如果 RKNN 的头文件和库不在系统路径下，可通过 `RKNN_API_PATH` 把 CMake 指向运行时包。

```shell
cd evaluate/rk3588_cpp
cmake -S . -B build -DRKNN_API_PATH=/path/to/rknn/runtime
cmake --build build -j
```

如果 `rknn_api.h` 与 `librknnrt.so` 已经安装在 `/usr/include` 与 `/usr/lib` 下，则可以省略 `-DRKNN_API_PATH=...`。

## 运行

```shell
./build/fear_rk3588_demo \
  --template_model ../../outputs/rk3588/fear_template_encoder.rknn \
  --track_model ../../outputs/rk3588/fear_track.rknn \
  --video ../../assets/test.mp4 \
  --output ../../outputs/rk3588_cpp/test.mp4 \
  --bbox 163,53,45,174 \
  --core auto
```

选项：

- `--bbox x,y,w,h`：首帧中要跟踪目标的初始框
- `--core auto|0|1|2|all`：RK3588 NPU 核心掩码

OpenCV 以 BGR 读取视频帧。该 demo 在做模型预处理前会把每帧转换为 RGB，与 Python 端跟踪器的归一化保持一致。
