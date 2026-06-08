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

## RK3588 deployment

RK3588 runs the tracker through Rockchip RKNN. The export path produces two models:

1. `fear_template_encoder.rknn`: first-frame template crop `[1,3,128,128]` to template features `[1,256,8,8]`
2. `fear_track.rknn`: search crop `[1,3,256,256]` plus template features to bbox/classification maps

Install the normal training/export dependencies from `requirements.txt`, then install Rockchip's `rknn-toolkit2` wheel on the export machine. Export ONNX and RKNN from the project root:

```shell
PYTHONPATH=. python evaluate/rk3588_export.py \
  --weights_path=evaluate/checkpoints/FEAR-XS-NoEmbs.ckpt \
  --output_dir=outputs/rk3588
```

`--weights_path` can point to `.ckpt`, `.pt`, or `.pth` files. The exporter supports:

- PyTorch Lightning checkpoints with `checkpoint["state_dict"]`
- plain `torch.save(model.state_dict(), "model.pt")` files
- checkpoint dictionaries with `model_state_dict`, `model`, `net`, or `module` keys
- complete `torch.save(model, "model.pt")` files

For a plain `.pt` state dict:

```shell
PYTHONPATH=. python evaluate/rk3588_export.py \
  --weights_path=/path/to/model.pt \
  --config_path=model_training/config/model/fear.yaml \
  --output_dir=outputs/rk3588
```

If the saved weights are compatible but do not exactly match every key in the configured model, retry with `--strict_weights=False`.

If `rknn-toolkit2` is not available on the export machine, generate only ONNX files:

```shell
PYTHONPATH=. python evaluate/rk3588_export.py --skip_rknn=True
```

For INT8 quantization, pass RKNN Toolkit2 calibration dataset files for both graphs:

```shell
PYTHONPATH=. python evaluate/rk3588_export.py \
  --do_quantization=True \
  --template_dataset=/path/to/template_dataset.txt \
  --track_dataset=/path/to/track_dataset.txt
```

Copy `outputs/rk3588/fear_template_encoder.rknn` and `outputs/rk3588/fear_track.rknn` to the RK3588 board. On the board, install the lightweight runtime dependencies from `requirements-rk3588.txt` and Rockchip's `rknn-toolkit-lite2` wheel, then run:

```shell
PYTHONPATH=. python evaluate/rk3588_demo_video.py \
  --template_model_path=outputs/rk3588/fear_template_encoder.rknn \
  --track_model_path=outputs/rk3588/fear_track.rknn \
  --initial_bbox='[163,53,45,174]' \
  --video_path=assets/test.mp4 \
  --output_path=outputs/rk3588/test.mp4
```

The RK3588 runtime tracker in `evaluate/rk3588_runtime.py` accepts RGB frames and `[x, y, width, height]` boxes, and only depends on `numpy`, `opencv-python-headless`, and `rknn-toolkit-lite2`.

A C++ RKNN C API demo for RK3588 is available in `evaluate/rk3588_cpp`. Build it on the board with CMake and run it with the same two `.rknn` files:

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
