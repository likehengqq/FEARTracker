<div align="center">

# FEAR: Fast, Efficient, Accurate and Robust Visual Tracker
[![Paper](https://img.shields.io/badge/arXiv-2112.07957-brightgreen)](https://arxiv.org/abs/2112.07957)
[![Conference](https://img.shields.io/badge/ECCV-2022-blue)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820625.pdf)

</div>

![FEAR architecture](./docs/architecture.png)

This is an official repository for the paper
```
FEAR: Fast, Efficient, Accurate and Robust Visual Tracker
Vasyl Borsuk, Roman Vei, Orest Kupyn, Tetiana Martyniuk, Igor Krashenyi, Jiři Matas
ECCV 2022
```

## Environment setup
The training code is tested on Linux systems and mobile benchmarking on MacOS systems.
```shell
conda create -n py37fear python=3.7
conda activate py37fear
pip install -r requirements.txt
```
**N.B.** You might need to remove `xtcocotools` requirement when installing environment on MacOS system for model evaluation.

## Demo inference with Python
The FEAR-XS model checkpoint is available in the `evaluate/checkpoints` folder. To run the inference code:
```shell
PYTHONPATH=. python demo_video.py --initial_bbox=[163,53,45,174] \
--video_path=assets/test.mp4 \
--output_path=outputs/test.mp4
```
**N.B.** This FEAR-XS model is releeased without the Dynamic Template Update.

## FEAR Benchmark
We provide FEAR evaluation protocol implementation in `evaluate/MeasurePerformance` directory. 
You should do the following steps on MacOS device to evaluate model on iOS device:
1. Open `evaluate/MeasurePerformance` project in Xcode. 
You can do this by double-clicking on `evaluate/MeasurePerformance/MeasurePerformance.xcodeproj` file or by opening it from the Open option in Xcode. 
2. Connect iOS device to your computer and build the project into it. 
3. Select one of the benchmark options by tapping on the corresponding button on your mobile device:
   - _Benchmark FPS_: launches simple model benchmark that warms up the model for 20 iterations and measures average FPS across 100 model calls. The result is displayed in Xcode console.
   - _Benchmark Online_: launches FEAR online benchmark as described in the paper
   - _Benchmark Offline_: launches FEAR offline benchmark

Do the following steps on MacOS device to convert model into CoreML:
1. To convert the model trained in PyTorch to CoreML with the following command from the project root directory.
This command will produce a file with the model in CoreML format (`Model.mlmodel`) and a model with FP16 weight quantization (`Model_quantized.mlmodel`).
 ```shell
 PYTHONPATH=. python evaluate/coreml_convert.py
 ```
2. Move converted model into the iOS project with the following command `cp Model_quantized.mlmodel evaluate/MeasurePerformance/MeasurePerformance/models/Model_quantized.mlmodel`.

## 在 RK3588 上部署

RK3588 通过瑞芯微（Rockchip）RKNN 运行跟踪器。导出流程会生成两个模型：

1. `fear_template_encoder.rknn`：首帧模板裁剪 `[1,3,128,128]` → 模板特征 `[1,256,8,8]`
2. `fear_track.rknn`：搜索裁剪 `[1,3,256,256]` + 模板特征 → bbox / 分类图

先按 `requirements.txt` 安装常规的训练/导出依赖，再在导出机器上安装瑞芯微的 `rknn-toolkit2` whl 包。在项目根目录导出 ONNX 与 RKNN：

```shell
PYTHONPATH=. python evaluate/rk3588_export.py \
  --weights_path=evaluate/checkpoints/FEAR-XS-NoEmbs.ckpt \
  --output_dir=outputs/rk3588
```

`--weights_path` 可以指向 `.ckpt`、`.pt` 或 `.pth` 文件。导出脚本支持以下权重格式：

- 含 `checkpoint["state_dict"]` 的 PyTorch Lightning checkpoint
- 直接用 `torch.save(model.state_dict(), "model.pt")` 保存的文件
- 含 `model_state_dict`、`model`、`net` 或 `module` 键的 checkpoint 字典
- 用 `torch.save(model, "model.pt")` 保存的完整模型文件

对于普通的 `.pt` state dict：

```shell
PYTHONPATH=. python evaluate/rk3588_export.py \
  --weights_path=/path/to/model.pt \
  --config_path=model_training/config/model/fear.yaml \
  --output_dir=outputs/rk3588
```

如果保存的权重兼容、但与所配置模型的每个键不完全匹配，可加 `--strict_weights=False` 重试。

如果导出机器上没有 `rknn-toolkit2`，可以只生成 ONNX 文件：

```shell
PYTHONPATH=. python evaluate/rk3588_export.py --skip_rknn=True
```

如需 INT8 量化，请为两个子图分别传入 RKNN Toolkit2 的校准数据集文件：

```shell
PYTHONPATH=. python evaluate/rk3588_export.py \
  --do_quantization=True \
  --template_dataset=/path/to/template_dataset.txt \
  --track_dataset=/path/to/track_dataset.txt
```

把 `outputs/rk3588/fear_template_encoder.rknn` 和 `outputs/rk3588/fear_track.rknn` 拷贝到 RK3588 开发板。在板上按 `requirements-rk3588.txt` 安装轻量运行时依赖以及瑞芯微的 `rknn-toolkit-lite2` whl 包，然后运行：

```shell
PYTHONPATH=. python evaluate/rk3588_demo_video.py \
  --template_model_path=outputs/rk3588/fear_template_encoder.rknn \
  --track_model_path=outputs/rk3588/fear_track.rknn \
  --initial_bbox='[163,53,45,174]' \
  --video_path=assets/test.mp4 \
  --output_path=outputs/rk3588/test.mp4
```

`evaluate/rk3588_runtime.py` 中的 RK3588 运行时跟踪器接受 RGB 帧与 `[x, y, width, height]` 格式的框，仅依赖 `numpy`、`opencv-python-headless` 和 `rknn-toolkit-lite2`。

`evaluate/rk3588_cpp` 提供了一个基于 RKNN C API 的 C++ demo。在板上用 CMake 编译，并使用同样的两个 `.rknn` 文件运行：

```shell
cd evaluate/rk3588_cpp
cmake -S . -B build -DRKNN_API_PATH=/path/to/rknn/runtime
cmake --build build -j
./build/fear_rk3588_demo \
  --template_model ../../outputs/rk3588/fear_template_encoder.rknn \
  --track_model ../../outputs/rk3588/fear_track.rknn \
  --video ../../assets/test.mp4 \
  --output ../../outputs/rk3588_cpp/test.mp4 \
  --bbox 163,53,45,174
```

### 性能基准与精度对齐

实测帧率与评估量化精度损失（RK3588 与 RV1126B 通用）：

```shell
# 真机分段计时 + 端到端 FPS（板端单核 NPU 用 core_mask=auto，RK3588 多核可用 all）
PYTHONPATH=. python evaluate/rk3588_benchmark.py \
  --template_model_path=outputs/rk3588/fear_template_encoder.rknn \
  --track_model_path=outputs/rk3588/fear_track.rknn \
  --video_path=assets/test.mp4 --core_mask=auto

# 转换前后逐输出精度对齐：PyTorch vs ONNX vs RKNN(模拟器/可选 INT8)
PYTHONPATH=. python evaluate/rk3588_accuracy.py \
  --template_onnx=outputs/rk3588/fear_template_encoder.onnx \
  --track_onnx=outputs/rk3588/fear_track.onnx
```

两个脚本都支持 `--backend=onnx`（基准脚本）/ 默认 ONNX 对比（精度脚本），便于在没有板子的 PC 上先验证流程。

> 在瑞芯微 **RV1126B** 上部署同样走 RKNN 工具链，可复用上面的导出/基准/精度脚本（导出时传 `--target_platform=rv1126b`）。详见
> [`docs/rv1126b_deployment.md`](docs/rv1126b_deployment.md)。

### Count FLOPS and parameters
```shell
PYTHONPATH=. python evaluate/macs_params.py
```

## Demo app for iOS

[Demo app screen recording](https://user-images.githubusercontent.com/24678253/179550055-689ee927-ff22-4c19-8087-539623cb1c2c.mp4)

1. Open `evaluate/FEARDemo` project in Xcode.
2. Connect iOS device to your computer and build the project. 
Make sure to enable developer mode on your iOS device and trust your current apple developer.
Also, you will need to select a development team under the signing & capabilities pane of the project editor (navigation described here [here](https://developer.apple.com/documentation/xcode/adding-capabilities-to-your-app))

**N.B.** The demo app does not contain bounding box smoothing postprocessing steps of the tracker so its output is slightly different from Python.

## Training
### Data preparation
There are two dataset configurations. 
Download all datasets from the configuration file you'll train with and put them into the directory specified in `visual_object_tracking_datasets` configuration field.
You can change the value of `visual_object_tracking_datasets` to your local dataset path.
There are two dataset configurations:
1. Quick train on GOT-10k dataset <br />
   Config file: `model_training/config/dataset/got10k_train.yaml`
2. Full train on LaSOT, COCO2017, YouTube-BoundingBoxes, GOT-10k and ILSVRC <br />
   Config file: `model_training/config/dataset/full_train.yaml`

You should create CSV annotation file for each of training datasets.
We don't provide CSV annotations as some datasets have license restrictions.
The annotation file for each dataset should have the following format:
- `sequence_id: str` - unique identifier of video file
- `track_id: str` - unique identifier of scene inside video file
- `frame_index: int` - index of frame inside video
- `img_path: str` - location of frame image relative to root folder with all datasets
- `bbox: Tuple[int, int, int, int]` - bounding box of object in a format `x, y, w, h`
- `frame_shape: Tuple[int, int]` - width and height of image
- `dataset: str` - label to identify dataset (example: `got10k`)
- `presence: int` - presence of the object (example, `0/1`)
- `near_corner: int` - is bounding box touches borders of the image (example, `0/1`)

### Run training
Current training code supports model training without Dynamic Template Update module, it'll be added soon.
You can launch training with default configuration with the following command from the project root directory:
```shell
PYTHONPATH=. python model_training/train.py backend=2gpu
# or the following for full train
PYTHONPATH=. python model_training/train.py dataset=full_train backend=2gpu
```

## Citation

If you use the FEAR Tracker benchmark, demo, models or code for your research projects, please cite the following paper:

```
@inproceedings{borsuk2022fear,
  title={FEAR: Fast, efficient, accurate and robust visual tracker},
  author={Borsuk, Vasyl and Vei, Roman and Kupyn, Orest and Martyniuk, Tetiana and Krashenyi, Igor and Matas, Ji{\v{r}}i},
  booktitle={European Conference on Computer Vision},
  pages={644--663},
  year={2022},
  organization={Springer}
}
```
