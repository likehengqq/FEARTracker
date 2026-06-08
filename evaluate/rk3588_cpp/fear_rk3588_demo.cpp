#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include "rknn_api.h"

namespace {
namespace fs = std::filesystem;

constexpr int kTemplateSize = 128;
constexpr int kInstanceSize = 256;
constexpr int kScoreSize = 16;
constexpr int kTotalStride = 16;
constexpr float kTemplateBBoxOffset = 0.2f;
constexpr float kSearchContext = 2.0f;
constexpr float kPenaltyK = 0.062f;
constexpr float kWindowInfluence = 0.38f;
constexpr float kLr = 0.765f;
constexpr bool kSmooth = false;

struct BBox {
  float x = 0.0f;
  float y = 0.0f;
  float w = 0.0f;
  float h = 0.0f;
};

struct CropResult {
  cv::Mat crop_rgb;
  BBox crop_bbox;
  BBox context;
};

struct RawTensor {
  std::vector<float> data;
  rknn_tensor_attr attr;
};

struct TensorNCHW {
  std::vector<float> data;
  int n = 1;
  int c = 0;
  int h = 0;
  int w = 0;

  float at(int channel, int row, int col) const {
    return data[(channel * h + row) * w + col];
  }
};

struct Args {
  std::string template_model = "outputs/rk3588/fear_template_encoder.rknn";
  std::string track_model = "outputs/rk3588/fear_track.rknn";
  std::string video_path = "assets/test.mp4";
  std::string output_path = "outputs/rk3588_cpp/test.mp4";
  std::string bbox = "163,53,45,174";
  std::string core = "auto";
};

std::vector<uint8_t> ReadFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("File is empty: " + path);
  }
  std::vector<uint8_t> buffer(static_cast<size_t>(size));
  file.seekg(0, std::ios::beg);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    throw std::runtime_error("Failed to read file: " + path);
  }
  return buffer;
}

void CheckRknn(int ret, const std::string& message) {
  if (ret != RKNN_SUCC) {
    throw std::runtime_error(message + ", ret=" + std::to_string(ret));
  }
}

rknn_core_mask ParseCoreMask(const std::string& core) {
  if (core == "auto") {
    return RKNN_NPU_CORE_AUTO;
  }
  if (core == "0") {
    return RKNN_NPU_CORE_0;
  }
  if (core == "1") {
    return RKNN_NPU_CORE_1;
  }
  if (core == "2") {
    return RKNN_NPU_CORE_2;
  }
  if (core == "all") {
    return RKNN_NPU_CORE_0_1_2;
  }
  throw std::runtime_error("Unsupported --core value: " + core + " (use auto, 0, 1, 2, all)");
}

std::string TensorDimsToString(const rknn_tensor_attr& attr) {
  std::string result = "[";
  for (uint32_t i = 0; i < attr.n_dims; ++i) {
    result += std::to_string(attr.dims[i]);
    if (i + 1 < attr.n_dims) {
      result += ",";
    }
  }
  result += "]";
  return result;
}

class RknnModel {
 public:
  RknnModel(const std::string& model_path, const std::string& core) : model_data_(ReadFile(model_path)) {
    CheckRknn(
        rknn_init(&ctx_, model_data_.data(), static_cast<uint32_t>(model_data_.size()), 0, nullptr),
        "rknn_init failed for " + model_path);
    CheckRknn(rknn_set_core_mask(ctx_, ParseCoreMask(core)), "rknn_set_core_mask failed for " + model_path);

    std::memset(&io_num_, 0, sizeof(io_num_));
    CheckRknn(rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_)), "query io num failed");

    input_attrs_.resize(io_num_.n_input);
    for (uint32_t i = 0; i < io_num_.n_input; ++i) {
      std::memset(&input_attrs_[i], 0, sizeof(rknn_tensor_attr));
      input_attrs_[i].index = i;
      CheckRknn(rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr)),
                "query input attr failed");
    }

    output_attrs_.resize(io_num_.n_output);
    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
      std::memset(&output_attrs_[i], 0, sizeof(rknn_tensor_attr));
      output_attrs_[i].index = i;
      CheckRknn(rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr)),
                "query output attr failed");
    }
  }

  RknnModel(const RknnModel&) = delete;
  RknnModel& operator=(const RknnModel&) = delete;

  ~RknnModel() {
    if (ctx_ != 0) {
      rknn_destroy(ctx_);
    }
  }

  std::vector<RawTensor> Infer(const std::vector<std::vector<float>>& inputs_data) {
    if (inputs_data.size() != input_attrs_.size()) {
      throw std::runtime_error("Input count mismatch: got " + std::to_string(inputs_data.size()) +
                               ", expected " + std::to_string(input_attrs_.size()));
    }

    std::vector<rknn_input> inputs(inputs_data.size());
    for (size_t i = 0; i < inputs_data.size(); ++i) {
      std::memset(&inputs[i], 0, sizeof(rknn_input));
      inputs[i].index = static_cast<uint32_t>(i);
      inputs[i].type = RKNN_TENSOR_FLOAT32;
      inputs[i].fmt = RKNN_TENSOR_NCHW;
      inputs[i].size = static_cast<uint32_t>(inputs_data[i].size() * sizeof(float));
      inputs[i].buf = const_cast<float*>(inputs_data[i].data());
      inputs[i].pass_through = 0;
    }

    CheckRknn(rknn_inputs_set(ctx_, static_cast<uint32_t>(inputs.size()), inputs.data()), "rknn_inputs_set failed");
    CheckRknn(rknn_run(ctx_, nullptr), "rknn_run failed");

    std::vector<rknn_output> outputs(output_attrs_.size());
    for (auto& output : outputs) {
      std::memset(&output, 0, sizeof(rknn_output));
      output.want_float = 1;
      output.is_prealloc = 0;
    }
    CheckRknn(rknn_outputs_get(ctx_, static_cast<uint32_t>(outputs.size()), outputs.data(), nullptr),
              "rknn_outputs_get failed");

    std::vector<RawTensor> result(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      const size_t numel = outputs[i].size / sizeof(float);
      const float* ptr = reinterpret_cast<const float*>(outputs[i].buf);
      result[i].data.assign(ptr, ptr + numel);
      result[i].attr = output_attrs_[i];
    }

    CheckRknn(rknn_outputs_release(ctx_, static_cast<uint32_t>(outputs.size()), outputs.data()),
              "rknn_outputs_release failed");
    return result;
  }

 private:
  std::vector<uint8_t> model_data_;
  rknn_context ctx_ = 0;
  rknn_input_output_num io_num_;
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;
};

int TruncToInt(float value) {
  return static_cast<int>(value);
}

BBox ExtendBBox(const BBox& bbox, float offset) {
  return BBox{
      static_cast<float>(TruncToInt(bbox.x - bbox.w * offset)),
      static_cast<float>(TruncToInt(bbox.y - bbox.h * offset)),
      static_cast<float>(TruncToInt(bbox.w * (1.0f + 2.0f * offset))),
      static_cast<float>(TruncToInt(bbox.h * (1.0f + 2.0f * offset))),
  };
}

BBox EnsureBBoxBoundaries(const BBox& bbox, const cv::Size& image_size) {
  const float x1 = std::min(std::max(0.0f, bbox.x), static_cast<float>(image_size.width));
  const float y1 = std::min(std::max(0.0f, bbox.y), static_cast<float>(image_size.height));
  const float x2 = std::min(std::max(0.0f, x1 + bbox.w), static_cast<float>(image_size.width));
  const float y2 = std::min(std::max(0.0f, y1 + bbox.h), static_cast<float>(image_size.height));
  return BBox{x1, y1, x2 - x1, y2 - y1};
}

BBox ClampBBox(const BBox& input_bbox, const cv::Size& image_size, int min_side = 3) {
  BBox bbox = EnsureBBoxBoundaries(input_bbox, image_size);
  if (bbox.w < min_side) {
    bbox.w = static_cast<float>(min_side);
    bbox.x -= std::max(0.0f, bbox.x + bbox.w - image_size.width);
  }
  if (bbox.h < min_side) {
    bbox.h = static_cast<float>(min_side);
    bbox.y -= std::max(0.0f, bbox.y + bbox.h - image_size.height);
  }
  return bbox;
}

cv::Scalar MeanColor(const cv::Mat& rgb) {
  return cv::mean(rgb);
}

BBox ResizeBBox(const BBox& bbox, const cv::Size& from_size, int out_size) {
  const float scale_x = static_cast<float>(out_size) / static_cast<float>(from_size.width);
  const float scale_y = static_cast<float>(out_size) / static_cast<float>(from_size.height);
  return BBox{bbox.x * scale_x, bbox.y * scale_y, bbox.w * scale_x, bbox.h * scale_y};
}

CropResult GetExtendedCrop(const cv::Mat& rgb, const BBox& bbox, int crop_size, float offset, cv::Scalar padding) {
  BBox context = ExtendBBox(bbox, offset);
  const int context_x = static_cast<int>(context.x);
  const int context_y = static_cast<int>(context.y);
  const int context_w = static_cast<int>(context.w);
  const int context_h = static_cast<int>(context.h);

  const int pad_left = std::max(-context_x, 0);
  const int pad_top = std::max(-context_y, 0);
  const int pad_right = std::max(context_x + context_w - rgb.cols, 0);
  const int pad_bottom = std::max(context_y + context_h - rgb.rows, 0);

  const int crop_x = context_x + pad_left;
  const int crop_y = context_y + pad_top;
  const int crop_w = context_w - pad_left - pad_right;
  const int crop_h = context_h - pad_top - pad_bottom;
  if (crop_w <= 0 || crop_h <= 0) {
    throw std::runtime_error("Invalid crop computed from bbox");
  }

  cv::Mat crop = rgb(cv::Rect(crop_x, crop_y, crop_w, crop_h));
  cv::Mat padded_crop;
  cv::copyMakeBorder(crop, padded_crop, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, padding);

  BBox padded_bbox{
      bbox.x - context.x,
      bbox.y - context.y,
      bbox.w,
      bbox.h,
  };
  padded_bbox = EnsureBBoxBoundaries(padded_bbox, padded_crop.size());
  BBox crop_bbox = ResizeBBox(padded_bbox, padded_crop.size(), crop_size);

  cv::Mat resized;
  cv::resize(padded_crop, resized, cv::Size(crop_size, crop_size), 0.0, 0.0, cv::INTER_LINEAR);
  return CropResult{resized, crop_bbox, context};
}

std::vector<float> PreprocessRgbToNchw(const cv::Mat& rgb) {
  cv::Mat rgb_float;
  rgb.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);

  const float mean[3] = {0.485f, 0.456f, 0.406f};
  const float std[3] = {0.229f, 0.224f, 0.225f};
  std::vector<float> nchw(3 * rgb.rows * rgb.cols);
  for (int y = 0; y < rgb.rows; ++y) {
    const cv::Vec3f* row = rgb_float.ptr<cv::Vec3f>(y);
    for (int x = 0; x < rgb.cols; ++x) {
      for (int c = 0; c < 3; ++c) {
        nchw[(c * rgb.rows + y) * rgb.cols + x] = (row[x][c] - mean[c]) / std[c];
      }
    }
  }
  return nchw;
}

std::pair<std::vector<float>, std::vector<float>> MakeGrid() {
  std::vector<float> grid_x(kScoreSize * kScoreSize);
  std::vector<float> grid_y(kScoreSize * kScoreSize);
  const float center = std::floor(static_cast<float>(kScoreSize / 2));
  for (int row = 0; row < kScoreSize; ++row) {
    for (int col = 0; col < kScoreSize; ++col) {
      const int idx = row * kScoreSize + col;
      grid_x[idx] = (static_cast<float>(col) - center) * kTotalStride + kInstanceSize / 2;
      grid_y[idx] = (static_cast<float>(row) - center) * kTotalStride + kInstanceSize / 2;
    }
  }
  return {grid_x, grid_y};
}

std::vector<float> MakeWindow() {
  std::vector<float> hanning(kScoreSize);
  for (int i = 0; i < kScoreSize; ++i) {
    hanning[i] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(CV_PI) * i / (kScoreSize - 1));
  }

  std::vector<float> window(kScoreSize * kScoreSize);
  for (int row = 0; row < kScoreSize; ++row) {
    for (int col = 0; col < kScoreSize; ++col) {
      window[row * kScoreSize + col] = hanning[row] * hanning[col];
    }
  }
  return window;
}

float Sigmoid(float value) {
  return 1.0f / (1.0f + std::exp(-value));
}

float Limit(float value) {
  return std::max(value, 1.0f / value);
}

float SquaredSize(float w, float h) {
  const float pad = (w + h) * 0.5f;
  return std::sqrt((w + pad) * (h + pad));
}

TensorNCHW ToNchw(const RawTensor& raw, int expected_channels) {
  const rknn_tensor_attr& attr = raw.attr;
  TensorNCHW tensor;
  tensor.n = 1;

  if (attr.n_dims == 4) {
    const int d0 = attr.dims[0];
    const int d1 = attr.dims[1];
    const int d2 = attr.dims[2];
    const int d3 = attr.dims[3];
    const bool looks_nhwc = attr.fmt == RKNN_TENSOR_NHWC || d3 == expected_channels;
    if (looks_nhwc) {
      tensor.n = d0;
      tensor.h = d1;
      tensor.w = d2;
      tensor.c = d3;
      tensor.data.assign(tensor.c * tensor.h * tensor.w, 0.0f);
      for (int h = 0; h < tensor.h; ++h) {
        for (int w = 0; w < tensor.w; ++w) {
          for (int c = 0; c < tensor.c; ++c) {
            tensor.data[(c * tensor.h + h) * tensor.w + w] = raw.data[(h * tensor.w + w) * tensor.c + c];
          }
        }
      }
    } else {
      tensor.n = d0;
      tensor.c = d1;
      tensor.h = d2;
      tensor.w = d3;
      tensor.data = raw.data;
    }
  } else if (attr.n_dims == 3 && expected_channels == 1) {
    tensor.n = attr.dims[0];
    tensor.c = 1;
    tensor.h = attr.dims[1];
    tensor.w = attr.dims[2];
    tensor.data = raw.data;
  } else {
    throw std::runtime_error("Unsupported RKNN output dims " + TensorDimsToString(attr));
  }

  if (tensor.n != 1 || tensor.c != expected_channels) {
    throw std::runtime_error("Unexpected RKNN output shape " + TensorDimsToString(attr));
  }
  return tensor;
}

class FearRk3588Tracker {
 public:
  FearRk3588Tracker(const std::string& template_model, const std::string& track_model, const std::string& core)
      : template_model_(template_model, core), track_model_(track_model, core) {
    auto grid = MakeGrid();
    grid_x_ = std::move(grid.first);
    grid_y_ = std::move(grid.second);
    window_ = MakeWindow();
  }

  void Initialize(const cv::Mat& rgb, BBox bbox) {
    bbox_ = ClampBBox(bbox, rgb.size());
    mean_color_ = MeanColor(rgb);

    CropResult template_crop = GetExtendedCrop(rgb, bbox_, kTemplateSize, kTemplateBBoxOffset, mean_color_);
    std::vector<float> template_input = PreprocessRgbToNchw(template_crop.crop_rgb);
    std::vector<RawTensor> outputs = template_model_.Infer({template_input});
    if (outputs.empty()) {
      throw std::runtime_error("Template model returned no outputs");
    }
    TensorNCHW template_features = ToNchw(outputs[0], 256);
    template_features_ = std::move(template_features.data);
  }

  std::pair<BBox, float> Update(const cv::Mat& rgb) {
    if (template_features_.empty()) {
      throw std::runtime_error("Tracker is not initialized");
    }

    CropResult search_crop = GetExtendedCrop(rgb, bbox_, kInstanceSize, kSearchContext, mean_color_);
    prev_size_ = cv::Size2f(search_crop.crop_bbox.w, search_crop.crop_bbox.h);
    auto result = Track(search_crop.crop_rgb);
    bbox_ = RescaleBBox(result.first, search_crop.context);
    bbox_ = ClampBBox(bbox_, rgb.size());
    return {bbox_, result.second};
  }

 private:
  std::pair<BBox, float> Track(const cv::Mat& search_rgb) {
    std::vector<float> search_input = PreprocessRgbToNchw(search_rgb);
    std::vector<RawTensor> outputs = track_model_.Infer({search_input, template_features_});
    if (outputs.size() != 2) {
      throw std::runtime_error("Track model returned " + std::to_string(outputs.size()) + " outputs, expected 2");
    }

    TensorNCHW regression = ToNchw(outputs[0], 4);
    TensorNCHW cls = ToNchw(outputs[1], 1);
    return Postprocess(regression, cls);
  }

  std::pair<BBox, float> Postprocess(const TensorNCHW& regression, const TensorNCHW& cls) {
    std::vector<float> cls_score(kScoreSize * kScoreSize);
    for (int i = 0; i < kScoreSize * kScoreSize; ++i) {
      cls_score[i] = Sigmoid(cls.data[i]);
    }

    std::vector<float> classification_map = cls_score;
    std::vector<float> penalty(kScoreSize * kScoreSize, 1.0f);
    if (kSmooth) {
      for (int row = 0; row < kScoreSize; ++row) {
        for (int col = 0; col < kScoreSize; ++col) {
          const int idx = row * kScoreSize + col;
          const float pred_w = regression.at(0, row, col) + regression.at(2, row, col);
          const float pred_h = regression.at(1, row, col) + regression.at(3, row, col);
          const float s_c = Limit(SquaredSize(pred_w, pred_h) / SquaredSize(prev_size_.width, prev_size_.height));
          const float r_c = Limit((prev_size_.width / prev_size_.height) / (pred_w / pred_h));
          penalty[idx] = std::exp(-(r_c * s_c - 1.0f) * kPenaltyK);
          classification_map[idx] =
              penalty[idx] * cls_score[idx] * (1.0f - kWindowInfluence) + window_[idx] * kWindowInfluence;
        }
      }
    }

    const auto max_iter = std::max_element(classification_map.begin(), classification_map.end());
    const int max_idx = static_cast<int>(std::distance(classification_map.begin(), max_iter));
    const int row = max_idx / kScoreSize;
    const int col = max_idx % kScoreSize;
    const int idx = row * kScoreSize + col;

    BBox bbox{
        grid_x_[idx] - regression.at(0, row, col),
        grid_y_[idx] - regression.at(1, row, col),
        regression.at(0, row, col) + regression.at(2, row, col),
        regression.at(1, row, col) + regression.at(3, row, col),
    };

    if (kSmooth) {
      const float lr = penalty[idx] * cls_score[idx] * kLr;
      const float pred_w = bbox.w * lr;
      const float pred_h = bbox.h * lr;
      const float prev_w = prev_size_.width * (1.0f - lr);
      const float prev_h = prev_size_.height * (1.0f - lr);
      bbox.w = prev_w + lr * (pred_w + prev_w);
      bbox.h = prev_h + lr * (pred_h + prev_h);
    }

    return {bbox, cls_score[idx]};
  }

  BBox RescaleBBox(const BBox& bbox, const BBox& context) const {
    const float w_scale = context.w / static_cast<float>(kInstanceSize);
    const float h_scale = context.h / static_cast<float>(kInstanceSize);
    return BBox{
        std::round(bbox.x * w_scale + context.x),
        std::round(bbox.y * h_scale + context.y),
        std::max(3.0f, std::round(bbox.w * w_scale)),
        std::max(3.0f, std::round(bbox.h * h_scale)),
    };
  }

  RknnModel template_model_;
  RknnModel track_model_;
  std::vector<float> template_features_;
  std::vector<float> grid_x_;
  std::vector<float> grid_y_;
  std::vector<float> window_;
  BBox bbox_;
  cv::Size2f prev_size_;
  cv::Scalar mean_color_;
};

BBox ParseBBox(const std::string& text) {
  std::vector<float> values;
  std::string token;
  for (char ch : text) {
    if (ch == ',' || ch == ' ') {
      if (!token.empty()) {
        values.push_back(std::stof(token));
        token.clear();
      }
    } else {
      token.push_back(ch);
    }
  }
  if (!token.empty()) {
    values.push_back(std::stof(token));
  }
  if (values.size() != 4) {
    throw std::runtime_error("--bbox must contain four numbers: x,y,w,h");
  }
  return BBox{values[0], values[1], values[2], values[3]};
}

void PrintUsage(const char* program) {
  std::cout << "Usage: " << program << " [options]\n"
            << "Options:\n"
            << "  --template_model PATH  RKNN template encoder model\n"
            << "  --track_model PATH     RKNN tracking model\n"
            << "  --video PATH           Input video path\n"
            << "  --output PATH          Output video path\n"
            << "  --bbox x,y,w,h         Initial bbox, default 163,53,45,174\n"
            << "  --core auto|0|1|2|all  RK3588 NPU core mask, default auto\n"
            << "  --help                 Show this help\n";
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help" || key == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    }
    if (i + 1 >= argc) {
      throw std::runtime_error("Missing value for argument: " + key);
    }
    const std::string value = argv[++i];
    if (key == "--template_model") {
      args.template_model = value;
    } else if (key == "--track_model") {
      args.track_model = value;
    } else if (key == "--video") {
      args.video_path = value;
    } else if (key == "--output") {
      args.output_path = value;
    } else if (key == "--bbox") {
      args.bbox = value;
    } else if (key == "--core") {
      args.core = value;
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const BBox initial_bbox = ParseBBox(args.bbox);

    cv::VideoCapture capture(args.video_path);
    if (!capture.isOpened()) {
      throw std::runtime_error("Failed to open input video: " + args.video_path);
    }

    const double fps = capture.get(cv::CAP_PROP_FPS) > 0.0 ? capture.get(cv::CAP_PROP_FPS) : 25.0;
    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    const fs::path output_path(args.output_path);
    if (!output_path.parent_path().empty()) {
      fs::create_directories(output_path.parent_path());
    }
    cv::VideoWriter writer(
        args.output_path,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        cv::Size(width, height));
    if (!writer.isOpened()) {
      throw std::runtime_error("Failed to open output video: " + args.output_path);
    }

    FearRk3588Tracker tracker(args.template_model, args.track_model, args.core);

    cv::Mat frame_bgr;
    if (!capture.read(frame_bgr)) {
      throw std::runtime_error("Input video has no frames: " + args.video_path);
    }

    cv::Mat frame_rgb;
    cv::cvtColor(frame_bgr, frame_rgb, cv::COLOR_BGR2RGB);
    tracker.Initialize(frame_rgb, initial_bbox);

    BBox current_bbox = initial_bbox;
    int frame_index = 0;
    while (true) {
      cv::rectangle(
          frame_bgr,
          cv::Rect(
              static_cast<int>(current_bbox.x),
              static_cast<int>(current_bbox.y),
              static_cast<int>(current_bbox.w),
              static_cast<int>(current_bbox.h)),
          cv::Scalar(0, 255, 0),
          5);
      writer.write(frame_bgr);

      if (!capture.read(frame_bgr)) {
        break;
      }
      cv::cvtColor(frame_bgr, frame_rgb, cv::COLOR_BGR2RGB);
      auto update = tracker.Update(frame_rgb);
      current_bbox = update.first;
      ++frame_index;
      if (frame_index % 30 == 0) {
        std::cout << "Processed " << frame_index << " frames, score=" << update.second << std::endl;
      }
    }

    std::cout << "Wrote output video to " << args.output_path << std::endl;
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "Error: " << exc.what() << std::endl;
    return 1;
  }
}
