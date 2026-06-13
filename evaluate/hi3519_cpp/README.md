# FEAR Hi3519（NNIE）C++ 板端实现

本目录提供 FEAR 跟踪器在海思 Hi3519A / Hi3519A V100 上的 C++ 板端实现，采用
**「NNIE 骨干 + CPU 头部」** 的切分方案（详见 `../../docs/hi3519_deployment.md`）。

## 组成

| 文件 | 作用 | 是否需要 SDK |
| --- | --- | --- |
| `fear_head.hpp` | CPU 头部段（BoxTower：互相关 + Exp + 卷积塔），纯标准库实现 | 否 |
| `test_fear_head.cpp` | 头部数值自检：与 PyTorch 参考逐元素比较 | 否 |
| `fear_hi3519_demo.cpp` | 完整板端 demo：NNIE 骨干 + CPU 头部 + OpenCV 视频 | NNIE 部分需 SVP SDK |
| `CMakeLists.txt` | 构建脚本 | — |

为什么这样切：NNIE 不支持互相关 `MatMul` 与 `Exp`，所以**骨干特征提取（256×256 卷积，计算量主要在这）放 NNIE**，**BoxTower 头部（作用在 16×16 小图，含 MatMul/Exp）放 ARM CPU**。

## 准备模型与权重

在 x86 主机用 Python 导出（见 `../hi3519_split_export.py` / `../hi3519_dump_head.py`）：

```shell
# 1) 导出两个 NNIE 骨干 ONNX（模板 8x8 / 搜索 16x16）
PYTHONPATH=. python evaluate/hi3519_split_export.py --output_dir=outputs/hi3519_split
# 2) 导出 CPU 头部权重（折叠 BN）+ 自检参考数据
PYTHONPATH=. python evaluate/hi3519_dump_head.py --output_dir=outputs/hi3519_split
```

然后把两个骨干 ONNX 用 `nnie_mapper` 转成 `.wk`（流程见 `../../docs/hi3519_deployment.md`），
得到 `fear_backbone_template.wk` 与 `fear_backbone_search.wk`；头部用 `hi3519_head_weights.bin`。

## 头部数值自检（任意 PC，无需 SDK）

```shell
cd evaluate/hi3519_cpp
cmake -S . -B build          # 若默认编译器缺少 libstdc++，加 -DCMAKE_CXX_COMPILER=g++
cmake --build build -j
./build/test_fear_head \
  ../../outputs/hi3519_split/hi3519_head_weights.bin \
  ../../outputs/hi3519_split/hi3519_head_ref.bin
# 期望输出: [PASS] C++ 头部与 PyTorch 数值一致
```

## 板端编译与运行（Hi3519）

NNIE 骨干段通过 HiSVP NNIE API 运行，**需要海思 SVP SDK**。`fear_hi3519_demo.cpp` 中的
`SvpNnieBackbone` 用 `#ifdef USE_HISVP` 包裹，并 include 用户提供的 `svp_nnie_backbone_impl.hpp`
（基于 SVP SDK 的 `SVP_NNIE_LoadModel` / `SVP_NNIE_Forward` 实现，可参考 `sample_svp_nnie_software`）。
不同 SVP 版本 API 略有差异，请对照你的 `HiSVP API 参考` 适配。

```shell
cd evaluate/hi3519_cpp
cmake -S . -B build -DUSE_HISVP=ON -DHISVP_API_PATH=/path/to/svp/sdk
cmake --build build -j
./build/fear_hi3519_demo \
  --template_wk fear_backbone_template.wk \
  --search_wk   fear_backbone_search.wk \
  --head_weights hi3519_head_weights.bin \
  --video test.mp4 --output out.mp4 \
  --bbox 163,53,45,174
```

> 说明：本仓库 CI 只编译并验证了 `test_fear_head`（CPU 头部数值正确）。`fear_hi3519_demo`
> 的 NNIE 骨干部分依赖海思 SVP SDK 与具体板端环境，需在板上完成集成与实测。
> OpenCV 以 BGR 读帧，demo 在预处理前转 RGB，与 Python/RKNN 版本归一化一致。
