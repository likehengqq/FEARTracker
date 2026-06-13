# RKNN 性能基准与精度对齐工具说明（RK3588 / RV1126B）

本文档说明仓库中两个用于 RKNN 部署的辅助脚本：

- `evaluate/rk3588_benchmark.py` —— **真机性能基准**：分段计时并给出端到端 FPS。
- `evaluate/rk3588_accuracy.py` —— **转换前后精度对齐**：对比 PyTorch / ONNX / RKNN 的输出。

两个脚本同时适用于 RK3588 与 RV1126B（以及其他走 RKNN 工具链的瑞芯微芯片），区别仅在于
传入的 `.rknn` 模型、目标平台与 `core_mask`。

> 前置条件：先用 `evaluate/rk3588_export.py` 导出 ONNX（必要时再转 `.rknn`）。
> 导出/精度脚本依赖 `onnx`、`onnxruntime`（`pip install onnx onnxruntime`）；
> 板端基准依赖 `rknn-toolkit-lite2`；RKNN 模拟器精度对比依赖 `rknn-toolkit2`。

---

## 1. 性能基准 `rk3588_benchmark.py`

### 1.1 作用

按 FEAR 的真实推理流程逐帧运行跟踪器，并把每帧耗时拆成四段：

| 阶段 | 说明 | 运行位置 |
| --- | --- | --- |
| 预处理 | 裁剪后做 RGB 归一化、转 NCHW | CPU |
| NPU 跟踪 | 跟踪分支 `.rknn` 推理 | NPU |
| 后处理 | sigmoid / 解码 / 平滑等 | CPU |
| 其它 | 搜索区域裁剪、bbox 还原等 | CPU |

另外单独报告**模板分支**（初始化时只跑一次）的耗时。输出**端到端 FPS** 与**仅 NPU 跟踪 FPS**。

### 1.2 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--template_model_path` | `outputs/rv1126b/fear_template_encoder.rknn` | 模板分支模型（rknn 后端用 `.rknn`，onnx 后端用 `.onnx`） |
| `--track_model_path` | `outputs/rv1126b/fear_track.rknn` | 跟踪分支模型 |
| `--video_path` | `assets/test.mp4` | 测试视频；设为 `None` 则用随机合成帧 |
| `--initial_bbox` | `[163,53,45,174]` | 首帧初始框 `[x,y,w,h]` |
| `--num_frames` | `200` | 计时帧数（视频不足会循环填充） |
| `--warmup` | `10` | 预热帧数（不计入统计） |
| `--core_mask` | `auto` | NPU 核心：`auto/0/1/2/all`。**RV1126B 单核用 `auto`**；RK3588 多核可用 `all` |
| `--backend` | `rknn` | `rknn`（板端真机）或 `onnx`（PC 自检，仅验证流程，不代表板端性能） |
| `--synth_height/--synth_width` | `720 / 1280` | 无视频时合成帧的分辨率 |

### 1.3 板端用法

```shell
# RV1126B（单核）
PYTHONPATH=. python evaluate/rk3588_benchmark.py \
  --template_model_path=outputs/rv1126b/fear_template_encoder.rknn \
  --track_model_path=outputs/rv1126b/fear_track.rknn \
  --video_path=assets/test.mp4 --num_frames=300 --core_mask=auto

# RK3588（可用多核）
PYTHONPATH=. python evaluate/rk3588_benchmark.py \
  --template_model_path=outputs/rk3588/fear_template_encoder.rknn \
  --track_model_path=outputs/rk3588/fear_track.rknn \
  --core_mask=all
```

### 1.4 PC 自检（不需要板子）

```shell
PYTHONPATH=. python evaluate/rk3588_benchmark.py --backend=onnx \
  --template_model_path=outputs/rv1126b/fear_template_encoder.onnx \
  --track_model_path=outputs/rv1126b/fear_track.onnx
```

`--backend=onnx` 会注入一个用 onnxruntime 实现的假 `rknnlite`，复用完全相同的 CPU 前后处理逻辑，
用来验证脚本与流水线是否正确。**此时“NPU 跟踪”一栏是 CPU 上 onnxruntime 的耗时，不能当作板端性能。**

### 1.5 输出示例与解读

```
================ FEAR 跟踪性能基准 ================
后端           : rknn
帧分辨率       : 1280x720
core_mask      : auto
计时帧数       : 300 (预热 10)
--------------------------------------------------
模板分支(一次) : 6.12 ms
每帧端到端     : 28.40 ms
  ├─ 预处理    : 4.10 ms (14%)
  ├─ NPU 跟踪  : 18.90 ms (67%)
  ├─ 后处理    : 1.20 ms (4%)
  └─ 其它      : 4.20 ms (15%)
--------------------------------------------------
端到端 FPS     : 35.2
仅 NPU 跟踪 FPS: 52.9
==================================================
```

解读要点：

- **端到端 FPS** 才是实际可达帧率；“仅 NPU 跟踪 FPS”只反映 NPU 那一段。
- 若 **NPU 段占比高** → 模型本身偏重，考虑 INT8 量化 / 精简模型 / 降低输入分辨率。
- 若 **预处理/其它（CPU）占比高** → 瓶颈在 CPU，考虑用 NEON/OpenCV 优化裁剪与归一化、减少内存拷贝。
- FEAR 大量使用 depthwise 卷积，NPU 利用率通常不高，实测帧率常明显低于按算力估的理论值，**以本脚本实测为准**。

---

## 2. 精度对齐 `rk3588_accuracy.py`

### 2.1 作用

用**同一组输入**分别跑 PyTorch（参考）、ONNX、RKNN（模拟器，可选 INT8 量化），
对每个输出报告**最大绝对误差**与**余弦相似度**，定位“导出 / 量化”是否引入明显误差。

判定阈值（脚本内）：余弦相似度 > 0.999 且最大绝对误差 < 1e-2 标记 `ok`，否则标记 `CHECK`。

### 2.2 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--config_path` | `model_training/config/model/fear.yaml` | 模型配置 |
| `--weights_path` | `evaluate/checkpoints/FEAR-XS-NoEmbs.ckpt` | 权重（参考输出用） |
| `--template_onnx` | `outputs/rv1126b/fear_template_encoder.onnx` | 模板分支 ONNX |
| `--track_onnx` | `outputs/rv1126b/fear_track.onnx` | 跟踪分支 ONNX |
| `--check_rknn` | `False` | 是否额外跑 RKNN 模拟器对比（需 `rknn-toolkit2`） |
| `--target_platform` | `rv1126b` | RKNN 目标平台 |
| `--do_quantization` | `False` | RKNN 是否做 INT8 量化（评估量化掉点时设 True） |
| `--template_dataset/--track_dataset` | `None` | INT8 量化校准集（见下） |
| `--seed` | `0` | 随机输入种子（保证可复现） |

### 2.3 用法

```shell
# 仅 PyTorch vs ONNX（任意 x86 主机可跑，验证导出正确性）
PYTHONPATH=. python evaluate/rk3588_accuracy.py \
  --template_onnx=outputs/rv1126b/fear_template_encoder.onnx \
  --track_onnx=outputs/rv1126b/fear_track.onnx

# 加 RKNN 模拟器 + INT8 量化（评估量化掉点，需 rknn-toolkit2）
PYTHONPATH=. python evaluate/rk3588_accuracy.py \
  --target_platform=rv1126b --check_rknn=True --do_quantization=True \
  --template_dataset=/path/to/template_dataset.txt \
  --track_dataset=/path/to/track_dataset.txt
```

### 2.4 输出示例

```
================ 模板分支 (template) ================
 PyTorch vs ONNX:
  [ok] template_features            最大绝对误差=6.437e-06  余弦相似度=1.000000
================ 跟踪分支 (track) ===================
 PyTorch vs ONNX:
  [ok] bbox                         最大绝对误差=9.918e-05  余弦相似度=1.000000
  [ok] cls                          最大绝对误差=8.821e-06  余弦相似度=1.000000
====================================================
```

- PyTorch↔ONNX 误差应非常小（1e-3 以内）；若偏大，检查 opset / 导出是否正确。
- 打开 `--do_quantization=True` 后，PyTorch↔RKNN(INT8) 的误差体现的是**量化损失**；若 `cls` 余弦相似度明显下降，往往导致跟踪框漂移，应扩充/优化校准集或对敏感层保留更高精度。

---

## 3. INT8 量化校准集准备

RKNN 的校准集是文本文件，每行一条样本：

- **模板分支**（单输入）：每行一个 128×128 图像路径。
- **跟踪分支**（双输入 `search` + `template_features`）：需按 RKNN Toolkit2 的多输入格式，每行同时给出
  256×256 搜索图与对应的模板特征（可先用模板分支跑出 `template_features` 并保存为 `.npy`）。

建议从**真实跟踪场景**采集图像，覆盖不同目标尺度、光照与背景，数量 50~200 张较合适。
预处理（RGB、`mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]`，先 `/255` 再标准化）
必须与导出/板端一致。

---

## 4. 相关文档

- 导出脚本与 RK3588 部署：见 `README.md` 的「在 RK3588 上部署」一节。
- RV1126B 部署：`docs/rv1126b_deployment.md`。
- 海思 Hi3519（NNIE）部署：`docs/hi3519_deployment.md`。
