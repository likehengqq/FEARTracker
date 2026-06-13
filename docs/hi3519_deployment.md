# 在海思 Hi3519（NNIE）上转换与部署 FEAR 跟踪器

本文档说明如何把 FEAR 视觉跟踪器从 PyTorch 转换并部署到海思 **Hi3519A / Hi3519A V100** 平台的 **NNIE**（Neural Network Inference Engine）加速引擎上运行。

整体链路与 iOS 的 CoreML 链路（`evaluate/coreml_convert.py`）类似，只是后端换成海思工具链：

```
PyTorch (.ckpt)  ──►  ONNX  ──►  (简化/转 Caffe)  ──►  nnie_mapper  ──►  .wk  ──►  板端 SVP/NNIE 推理
                    (本仓库脚本)                      (海思工具链)
```

> 仓库内提供的脚本：`evaluate/hi3519_convert.py`，负责完成链路第一步（导出 ONNX）。后续转 Caffe、量化、生成 `.wk` 依赖海思私有 SDK，无法在本仓库环境内执行，本文给出完整命令与配置说明。

---

## 1. 模型结构与两阶段推理

FEAR 跟踪推理分为两个阶段（与 `demo_video.py` 一致）：

| 阶段 | 何时运行 | 输入 | 输出 |
| --- | --- | --- | --- |
| **模板分支** `get_features` | 跟踪初始化时运行一次 | `template [1,3,128,128]` | `template_features [1,256,8,8]` |
| **跟踪分支** `track` | 每帧运行 | `search [1,3,256,256]` + `template_features [1,256,8,8]` | `bbox [1,4,16,16]`、`cls [1,1,16,16]` |

导出脚本会分别导出这两个子图（`fear_template.onnx` / `fear_track.onnx`）。把模板分支单独拆出，可让板端只在初始化时跑一次模板特征提取，每帧只跑跟踪分支，省算力。

---

## 2. 转换环境准备

转换涉及两套环境：**主机导出环境**（跑 PyTorch / ONNX）与**海思工具链环境**（跑 nnie_mapper / RuyiStudio）。

### 2.1 主机导出环境（本仓库）

与本仓库开发环境一致（Python 3.7 + 仓库 `requirements.txt`），额外需要 `onnx`、`onnxruntime`（用于导出和数值校验）：

```shell
# 已按 README 建好 py37fear 环境后：
pip install onnx==1.8.1 onnxruntime==1.8.1
# 简化 ONNX 计算图（强烈建议，便于后续转 Caffe）：
pip install onnx-simplifier==0.3.10
```

> 说明：本仓库的 `requirements.txt` 不含 `onnx`，因为它只在 Hi3519 / 移动端导出时才需要。`torch==1.7.1` 默认导出 **opset ≤ 11**，与海思工具链 / `onnx2caffe` 的兼容性最好。

### 2.2 海思 NNIE 工具链环境

NNIE 模型转换由海思私有 SDK 提供，常见两种方式（任选其一）：

| 方式 | 工具 | 运行平台 | 说明 |
| --- | --- | --- | --- |
| 图形界面 | **RuyiStudio**（睿易工作室） | Windows / Ubuntu 16.04 | 海思官方 IDE，封装 `nnie_mapper`，可可视化配置量化与查看每层数据 |
| 命令行 | **nnie_mapper**（`nnie_mapper_12` 等） | Ubuntu 16.04 x86_64 | 位于 SVP SDK 的 `tools/nnie/linux/mapper`，吃 Caffe 模型 + `.cfg`，输出 `.wk` |

必备组件：

- **HiSVP（Smart Vision Platform）SDK**：与你的芯片/固件版本严格对应（例如 `Hi3519AV100R001SPCxxx`）。NNIE 的 `.wk` 与运行时库版本强绑定，务必版本一致。
- **Caffe 1.0**（带海思补丁版）：nnie_mapper 的标准输入是 Caffe `prototxt + caffemodel`。
- **onnx2caffe / onnx-simplifier**：把本脚本导出的 ONNX 转成 Caffe（社区工具，海思官方主要支持 Caffe）。
- **交叉编译工具链**：`arm-himix200-linux`（Hi3519A V100）用于编译板端推理程序。

> 版本提示：NNIE 版本随 SDK 而定（Hi3519A V100 通常为 **NNIE 1.x**，仅支持 Caffe 1.0 算子集合与固定输入尺寸）。请以你拿到的 SVP SDK 文档（`HiSVP API 参考` / `HiSVP 开发指南`）为准。

---

## 3. 第一步：导出 ONNX（本仓库脚本）

```shell
PYTHONPATH=. python evaluate/hi3519_convert.py \
    --weights_path=evaluate/checkpoints/FEAR-XS-NoEmbs.ckpt \
    --output_dir=outputs/hi3519 \
    --opset=11 \
    --verify=True
```

产物（位于 `outputs/hi3519/`）：

- `fear_template.onnx` —— 模板分支（纯卷积，NNIE 友好）
- `fear_track.onnx` —— 跟踪分支（含互相关与 exp，见第 6 节）
- `fear_hi3519_meta.json` —— 输入输出形状、预处理（mean/std）等元信息

脚本会用 `onnxruntime` 对每个子图做数值校验，PyTorch 与 ONNX 输出的最大绝对误差应在 `1e-3` 以内。

随后建议先做计算图简化（消除冗余的 `Shape/Gather/Unsqueeze` 等动态形状算子，对转 Caffe / NNIE 很关键）：

```shell
python -m onnxsim outputs/hi3519/fear_template.onnx outputs/hi3519/fear_template_sim.onnx
python -m onnxsim outputs/hi3519/fear_track.onnx    outputs/hi3519/fear_track_sim.onnx
```

---

## 4. 第二步：ONNX → Caffe

海思 `nnie_mapper` 的标准输入为 Caffe 1.0。使用 `onnx2caffe` 转换简化后的 ONNX：

```shell
# 以 onnx2caffe 为例（https://github.com/MTlab/onnx2caffe）
python convertCaffe.py outputs/hi3519/fear_template_sim.onnx \
    fear_template.prototxt fear_template.caffemodel
```

注意事项：

- **固定输入尺寸**：NNIE 不支持动态 shape，`prototxt` 的 `input_dim` 必须写死（模板 `1 3 128 128`，跟踪 `1 3 256 256` 与 `1 256 8 8`）。
- **深度可分离卷积**：`SepConv` 的 depthwise 卷积在 Caffe 中对应 `group == in_channels` 的 `Convolution`，NNIE 支持。
- 转换后务必用 Caffe（或 onnxruntime 对齐）抽查若干层输出，确认转换无误再进 nnie_mapper。

---

## 5. 第三步：nnie_mapper 量化生成 `.wk`

### 5.1 准备量化校准集

NNIE 采用定点量化（INT8/INT16），需要一批**有代表性的真实图像**做校准。建议从实际跟踪场景中采集图像，分别裁剪成模板分支（128×128）与跟踪分支（256×256）输入大小，各 50~200 张，生成 `image_list.txt`（每行一个图像路径）。

### 5.2 编写 mapper 配置（`.cfg`）

`nnie_mapper` 通过 `.cfg` 指定输入模型、量化模式与归一化。关键字段示例（以跟踪分支为例，具体字段名以你的 SDK 版本为准）：

```ini
[prototxt_file] ./fear_track.prototxt
[caffemodel_file] ./fear_track.caffemodel
[net_type] 0
[image_list] ./image_list_256.txt
[image_type] 1            ; RGB/BGR planar
[norm_type] 5             ; 带 mean 文件 + data_scale 的归一化
[mean_file] ./fear_mean.txt
[data_scale] 0.0039216    ; 见下方归一化换算
[compile_mode] 1          ; 0:INT8(高压缩)  1:INT16(更高精度，建议先用)
[instruction_name] fear_track_inst
[RGB_order] RGB
```

**归一化换算（重要）**：训练/推理端预处理为 `x' = (x/255 - mean) / std`（见 `model_training/tracker/base_tracker.py`，`mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]`）。NNIE 的归一化形式为 `x' = (x - mean_chn) * data_scale`。由于 NNIE 的 `data_scale` 通常为标量，per-channel 的 `std` 需做近似或借助 `mean_file` 配合：

- `mean_chn = mean * 255`，即 `[123.7, 116.3, 103.5]`，写入 `mean_file`；
- `data_scale ≈ 1 / (mean(std) * 255) ≈ 1 / (0.226 * 255) ≈ 0.01735`；
- 若需严格 per-channel 标准差，建议把 `1/std` 折叠进网络首层卷积权重，再让 NNIE 只做减均值。`fear_hi3519_meta.json` 中记录了精确的 mean/std 供换算。

### 5.3 执行转换

```shell
./nnie_mapper_12 fear_track.cfg     # 生成 fear_track_inst.wk
./nnie_mapper_12 fear_template.cfg  # 生成 fear_template_inst.wk
```

RuyiStudio 中则是新建 NNIE 工程、导入 prototxt/caffemodel、配置上述参数后点击「编译」，并可在「向量对比」里逐层查看量化误差。

---

## 6. 算子兼容性与网络拆分策略（关键）

用 `onnx` 查看导出图的算子分布可知：

- **模板分支** `fear_template.onnx`：仅含 `Conv / Relu / Add` → **可整图跑在 NNIE 上**。
- **跟踪分支** `fear_track.onnx`：主体是 `Conv / Relu`，但额外含：
  - `MatMul`（来自 `MobileCorrelation` 的互相关 `z·x`）；
  - `Exp`（来自 `BoxTower` 对 bbox 回归图的 `torch.exp`）；
  - `Shape / Gather / Reshape / Transpose / Unsqueeze`（互相关中的 `view/permute` 动态形状算子）。

NNIE 1.x **不支持任意 MatMul 与 Exp**，动态 reshape 也不友好。推荐采用 **NNIE 卷积主体 + CPU 后处理** 的分段方案：

1. **放到 NNIE**：模板分支整图；跟踪分支中两个特征编码（`cls_encode` / `reg_encode`）及其后的 `bbox_tower / cls_tower / bbox_pred / cls_pred` 卷积塔（互相关之后的部分也可在 ARM 端用小算子完成）。
2. **放到 ARM/DSP（CPU 段）**：
   - 互相关 `MatMul`（`MobileCorrelation`）——数据量小（`[1,256,8,8]` × `[1,256,16,16]`），ARM 上开销可忽略；
   - 最后的 `x = exp(adjust * bbox_pred + bias)` 与 `cls = 0.1 * cls_pred`；
   - 检测框解码 / 后处理（对应 `FEARTracker._postprocess`，本就在 CPU 上）。

实现方式有两种：

- **方式 A（推荐，最稳）**：在导出阶段进一步把 `fear_track` 拆成「互相关前的特征图」「互相关后的卷积塔」两个 ONNX，分别上 NNIE；中间的 matmul 与最后的 exp 用手写 C/NEON 在板端完成。可基于本脚本的 `TrackBranch` 自行细分子模块导出。
- **方式 B**：使用 NNIE 的「自定义层 / CPU 段（`SVP_NNIE_Forward` 多段）」机制，把不支持的算子声明为 CPU 段，由 NNIE 运行时在 CPU 与 NNIE 之间切换执行。

> 经验：`Exp` 仅作用在 4 通道 16×16 的回归图上，`MatMul` 也只在 8×8 / 16×16 小特征上，放到 ARM 端对帧率影响很小，却能保证 NNIE 段全部为标准卷积、量化稳定。

---

## 7. 板端推理集成

在 SVP/NNIE 运行时（参考 SDK 的 `sample_svp_nnie_software` 示例）按如下流程集成：

1. **初始化**：`SVP_NNIE_Forward` 加载 `fear_template_inst.wk`，对首帧目标模板（128×128）跑一次，得到 `template_features`，缓存在内存。
2. **每帧跟踪**：
   - 按目标上一帧位置裁剪 256×256 搜索区域（对应 `get_extended_crop`）；
   - 用 `fear_track_inst.wk` 跑 NNIE 卷积段，得到中间特征；
   - 在 ARM 端补齐互相关 `MatMul` 与 `exp`，得到 `bbox`/`cls`；
   - 复用 CPU 后处理（`box_coder.decode` + 平滑），输出最终 `[x,y,w,h]`。
3. 注意 NNIE 输入为 **planar RGB**、定点格式，喂数据前需按第 5.2 节的归一化参数预处理。

---

## 8. 精度验证与常见问题

- **逐层比对**：先用 `onnxruntime` 确认 ONNX 与 PyTorch 一致（脚本已自动做），再用 RuyiStudio 的向量对比确认量化前后误差；INT16 通常精度损失很小，INT8 若掉点明显可对敏感层保留 INT16。
- **动态 shape 报错**：务必先跑 `onnxsim` 并确认 `prototxt` 输入维度写死。
- **算子不支持报错**：基本来自 `MatMul / Exp / Reshape`，按第 6 节拆分到 CPU 段。
- **颜色通道 / 归一化**：FEAR 用 RGB，`mean/std` 见 `fear_hi3519_meta.json`；通道顺序或归一化配错会导致跟踪框乱跳。
- **版本不匹配**：`.wk` 必须与板端 SVP 运行时库、芯片固件版本一致，否则加载失败。

---

## 9. 参考

- 仓库脚本：`evaluate/hi3519_convert.py`（ONNX 导出）、`evaluate/coreml_convert.py`（iOS CoreML 链路，可类比）。
- 海思文档：`HiSVP 开发指南`、`HiSVP API 参考`、RuyiStudio 用户指南（随 SVP SDK 提供）。
- 社区工具：[onnx-simplifier](https://github.com/daquexian/onnx-simplifier)、[onnx2caffe](https://github.com/MTlab/onnx2caffe)。
