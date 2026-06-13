# 在瑞芯微 RV1126B 上转换与部署 FEAR 跟踪器（中文说明）

本文档说明如何把 FEAR 视觉跟踪器从 PyTorch 转换并部署到瑞芯微（Rockchip）**RV1126B** 芯片上运行。

RV1126B 与 RK3588 一样使用瑞芯微的 **RKNN / RKNPU** 推理栈，工具链完全相同
（`rknn-toolkit2` 导出 → `.rknn` → 板端 `rknn_api` / `rknn-toolkit-lite2`）。
因此 **本仓库为 RK3588 准备的脚本可直接复用**，只需把目标平台切换为 `rv1126b`。

> 注意：本文针对的是较新的 **RV1126B**（RKNPU，走 `rknn-toolkit2`），而不是更早的 RV1126
> （走旧的 `rknn-toolkit` v1）。请以你拿到的 `rknn-toolkit2` 版本的「支持平台列表」为准，
> 确认其包含 `rv1126b`。

整体链路：

```
PyTorch (.ckpt) ──► ONNX ──► rknn-toolkit2 (target_platform=rv1126b, INT8 量化) ──► .rknn ──► 板端 RKNPU 推理
        └──────────── evaluate/rk3588_export.py（复用） ────────────┘
```

---

## 1. 模型结构与两阶段推理

与 RK3588 / Hi3519 一致，FEAR 跟踪推理分两个阶段，导出为两个子图：

| 阶段 | 何时运行 | 输入 | 输出 |
| --- | --- | --- | --- |
| **模板编码器** | 跟踪初始化时运行一次 | `template [1,3,128,128]` | `template_features [1,256,8,8]` |
| **跟踪头** | 每帧运行 | `search [1,3,256,256]` + `template_features [1,256,8,8]` | `bbox [1,4,16,16]`、`cls [1,1,16,16]` |

对应导出文件：`fear_template_encoder.rknn` 与 `fear_track.rknn`。

---

## 2. 转换环境准备

涉及两套环境：**x86_64 导出主机** 与 **RV1126B 板端运行时**。

### 2.1 导出主机（x86_64）

- 本仓库的常规依赖：按 README 建好 Python 环境后 `pip install -r requirements.txt`。
- ONNX 导出额外依赖：`pip install onnx onnxruntime`（仅导出/校验用，不在核心 `requirements.txt` 中）。
- 瑞芯微 **`rknn-toolkit2`** whl 包（用于把 ONNX 转成 `.rknn` 并做量化）。请使用支持 `rv1126b`
  的版本，从瑞芯微官方渠道（如 Rockchip 的 rknn-toolkit2 仓库 / 资料包）获取与你板端固件匹配的版本。

### 2.2 RV1126B 板端

- 轻量运行时依赖：`pip install -r requirements-rk3588.txt`（`numpy` / `opencv-python-headless` 等，RV1126B 通用）。
- 板端推理库二选一：
  - Python：瑞芯微 **`rknn-toolkit-lite2`**（对应 RV1126B 的 whl）；
  - C/C++：瑞芯微 **`librknnrt.so`** + `rknn_api.h`（对应 RV1126B 的运行时包）。
- 交叉编译工具链（编译 C++ demo 时需要），如 `arm-rockchip-linux-gnueabihf` / 对应 ABI 的工具链。

> 版本提示：`.rknn` 模型、板端 `librknnrt.so` / `rknn-toolkit-lite2` 与导出端 `rknn-toolkit2`
> 的大版本应保持一致，否则可能加载失败或精度异常。

---

## 3. 第一步：导出 ONNX + RKNN（复用 RK3588 脚本）

直接复用 `evaluate/rk3588_export.py`，把目标平台设为 `rv1126b`：

```shell
# 一步导出 ONNX 与 RKNN（需要 rknn-toolkit2）
PYTHONPATH=. python evaluate/rk3588_export.py \
  --weights_path=evaluate/checkpoints/FEAR-XS-NoEmbs.ckpt \
  --target_platform=rv1126b \
  --output_dir=outputs/rv1126b
```

如果导出主机暂时没有 `rknn-toolkit2`，可以先只导出 ONNX，之后再转 RKNN：

```shell
PYTHONPATH=. python evaluate/rk3588_export.py --skip_rknn=True --output_dir=outputs/rv1126b
```

产物（位于 `outputs/rv1126b/`）：`fear_template_encoder.onnx` / `fear_track.onnx`，
以及（未跳过 RKNN 时）`fear_template_encoder.rknn` / `fear_track.rknn`。

`--weights_path` 支持 `.ckpt` / `.pt` / `.pth` 等多种格式（详见 README 的 RK3588 一节）。

---

## 4. 第二步：INT8 量化（RV1126B 强烈建议）

RV1126B 的算力与内存比 RK3588 更受限，**强烈建议做 INT8 量化**（而不是用 FP16/混合精度跑），
以获得可用的帧率与更小的内存占用。量化需要一批有代表性的真实校准图像：

1. 从实际跟踪场景采集图像，分别裁剪成模板分支（128×128）与跟踪分支（256×256）输入大小；
2. 生成 RKNN Toolkit2 的数据集文本文件。**跟踪图有两个输入**，每行需按 Toolkit2 的多输入格式
   同时给出 `search` 与 `template_features`（模板特征可先用模板分支跑出来再保存为 npy）。

```shell
PYTHONPATH=. python evaluate/rk3588_export.py \
  --target_platform=rv1126b \
  --do_quantization=True \
  --template_dataset=/path/to/template_dataset.txt \
  --track_dataset=/path/to/track_dataset.txt \
  --output_dir=outputs/rv1126b
```

预处理（RGB、`mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]`，先 `x/255` 再标准化）
与 RK3588 一致；量化与板端推理都必须使用同一套归一化参数。

---

## 5. 第三步：板端推理

把 `outputs/rv1126b/fear_template_encoder.rknn` 与 `outputs/rv1126b/fear_track.rknn` 拷贝到 RV1126B 板上。

### 5.1 Python（rknn-toolkit-lite2）

直接复用 `evaluate/rk3588_demo_video.py` / `evaluate/rk3588_runtime.py`。
**RV1126B 是单核 NPU**，`core_mask` 用默认的 `auto` 即可（不要用 RK3588 的多核 `all`）：

```shell
PYTHONPATH=. python evaluate/rk3588_demo_video.py \
  --template_model_path=outputs/rv1126b/fear_template_encoder.rknn \
  --track_model_path=outputs/rv1126b/fear_track.rknn \
  --initial_bbox='[163,53,45,174]' \
  --video_path=assets/test.mp4 \
  --output_path=outputs/rv1126b/test.mp4 \
  --core_mask=auto
```

`evaluate/rk3588_runtime.py` 中的 `RK3588FEARTracker` 仅依赖 `numpy` / `opencv` / `rknn-toolkit-lite2`，
接受 RGB 帧与 `[x, y, w, h]` 框，可直接用于 RV1126B。

### 5.2 C++（rknn_api / librknnrt）

复用 `evaluate/rk3588_cpp`，在板上用对应 RV1126B 的 `librknnrt.so` 编译运行：

```shell
cd evaluate/rk3588_cpp
cmake -S . -B build -DRKNN_API_PATH=/path/to/rv1126b/rknn/runtime
cmake --build build -j
./build/fear_rk3588_demo \
  --template_model ../../outputs/rv1126b/fear_template_encoder.rknn \
  --track_model ../../outputs/rv1126b/fear_track.rknn \
  --video ../../assets/test.mp4 \
  --output ../../outputs/rv1126b_cpp/test.mp4 \
  --bbox 163,53,45,174 \
  --core auto
```

> C++ demo 名为 `fear_rk3588_demo`，但其逻辑与平台无关，RV1126B 上同样适用，只需链接 RV1126B 的运行时库。

---

## 6. 算子兼容性与精度说明

- 与海思 NNIE 不同，RKNN 对 `MatMul`（互相关）与 `Exp` 等算子有更完善的支持，
  `rknn-toolkit2` 在转换时通常能自动处理整图，一般无需手动把后处理拆到 CPU。
- 若转换时报某算子不支持，可：
  - 升级 `rknn-toolkit2` 到支持 `rv1126b` 的较新版本；
  - 或参考 `docs/hi3519_deployment.md` 的拆分思路，把互相关 / `exp` 放到 ARM 端，
    `.rknn` 只跑卷积主体。
- **精度验证流程**：先用 `onnxruntime` 确认 ONNX 与 PyTorch 一致（误差应 < 1e-3），
  再用 RKNN 的 `accuracy_analysis` / 模拟器逐层比对量化前后误差。INT8 若个别场景掉点明显，
  可对敏感层保留更高精度或扩充/优化校准集。
- **通道顺序 / 归一化** 配错是最常见问题，会导致跟踪框乱跳，务必保证板端预处理与导出时一致（RGB + 上述 mean/std）。

---

## 6.5 性能基准与精度对齐脚本

仓库提供两个脚本帮助你在 RV1126B 上**实测帧率**并**量化评估精度损失**：

### 真机性能基准 `evaluate/rk3588_benchmark.py`

分段计时（模板分支一次 / 每帧预处理 / NPU 跟踪推理 / CPU 后处理 / 其它），输出端到端 FPS 与各段占比。
RV1126B 是单核 NPU，`core_mask` 用 `auto`：

```shell
PYTHONPATH=. python evaluate/rk3588_benchmark.py \
  --template_model_path=outputs/rv1126b/fear_template_encoder.rknn \
  --track_model_path=outputs/rv1126b/fear_track.rknn \
  --video_path=assets/test.mp4 --num_frames=300 --core_mask=auto
```

> 这是回答“RV1126B 上能跑多少 FPS”的**唯一可靠方式**——必须在真机上实测。脚本也支持
> `--backend=onnx`（用 onnxruntime），但那只是用来在 PC 上验证脚本流程，**不代表板端性能**。
> 经验上 FEAR 用了大量 depthwise 卷积，NPU 利用率通常不高，且每帧的 CPU 预处理/后处理常成为瓶颈，
> 实测帧率往往明显低于按算力估的理论值。

### 转换前后精度对齐 `evaluate/rk3588_accuracy.py`

用同一组输入对比 PyTorch / ONNX / RKNN（模拟器，含可选 INT8 量化）的输出，逐输出报告
最大绝对误差与余弦相似度，用于判断导出与量化是否引入明显误差：

```shell
# 仅 PyTorch vs ONNX（任意主机可跑）
PYTHONPATH=. python evaluate/rk3588_accuracy.py \
  --template_onnx=outputs/rv1126b/fear_template_encoder.onnx \
  --track_onnx=outputs/rv1126b/fear_track.onnx

# 加 RKNN 模拟器 + INT8 量化对比（需 rknn-toolkit2）
PYTHONPATH=. python evaluate/rk3588_accuracy.py \
  --target_platform=rv1126b --check_rknn=True --do_quantization=True \
  --template_dataset=/path/to/template_dataset.txt \
  --track_dataset=/path/to/track_dataset.txt
```

---

## 7. RV1126B 与 RK3588 的主要差异

| 项目 | RK3588 | RV1126B |
| --- | --- | --- |
| 导出脚本 | `evaluate/rk3588_export.py` | 同一脚本，`--target_platform=rv1126b` |
| 量化 | 可选（算力较强） | **强烈建议 INT8** |
| NPU 核心 | 多核，`core_mask` 可用 `all` | 单核，`core_mask` 用 `auto` |
| 运行时库 | RK3588 版 `librknnrt` / `rknn-toolkit-lite2` | RV1126B 版对应库 |
| 板端脚本 / C++ demo | `rk3588_demo_video.py` / `rk3588_cpp` | 同样复用 |

---

## 8. 参考

- 复用脚本：`evaluate/rk3588_export.py`、`evaluate/rk3588_runtime.py`、`evaluate/rk3588_demo_video.py`、`evaluate/rk3588_cpp/`。
- 相关文档：`docs/hi3519_deployment.md`（海思 NNIE，含算子拆分思路）、README 的「在 RK3588 上部署」一节。
- 瑞芯微资料：`rknn-toolkit2` 用户指南、RKNPU 运行时（`librknnrt`）文档（请以官方与你板端固件匹配的版本为准）。
