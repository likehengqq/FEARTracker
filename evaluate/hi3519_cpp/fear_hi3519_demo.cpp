// FEAR Hi3519（NNIE）板端 C++ 推理 demo。
//
// 数据流（与 evaluate/hi3519_runtime.py 一致）：
//   首帧:  模板裁剪[128] --(NNIE 模板骨干 .wk)--> template_features[256,8,8]
//   每帧:  搜索裁剪[256] --(NNIE 搜索骨干 .wk)--> search_features[256,16,16]
//          (search_features, template_features) --(CPU 头部 FearHead)--> bbox/cls
//          CPU 解码 --> [x,y,w,h]
//
// 三部分：
//   - NNIE 骨干段（两个 .wk）：通过 HiSVP NNIE API 运行。**需要海思 SVP SDK**，
//     下面 SvpNnieBackbone 用 #ifdef USE_HISVP 包裹；不同 SVP 版本的结构体/函数名略有差异，
//     需对照你的 `HiSVP API 参考` 与 sample_comm_nnie 适配。本仓库 CI 未编译该部分。
//   - CPU 头部段：fear_head.hpp（已用 test_fear_head.cpp 验证与 PyTorch 数值一致）。
//   - OpenCV：视频读写、裁剪、缩放、padding、颜色转换、画框。
//
// 编译见 CMakeLists.txt（板端加 -DUSE_HISVP=ON 并提供 SVP SDK 路径）。
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "fear_head.hpp"

namespace {

constexpr int kTemplateSize = 128;
constexpr int kInstanceSize = 256;
constexpr int kScoreSize = 16;
constexpr int kTotalStride = 16;
constexpr float kTemplateBBoxOffset = 0.2f;
constexpr float kSearchContext = 2.0f;
const float kMean[3] = {0.485f, 0.456f, 0.406f};
const float kStd[3] = {0.229f, 0.224f, 0.225f};

struct BBox { float x = 0, y = 0, w = 0, h = 0; };

// ----------- NNIE 骨干抽象接口 -----------
// run(): 输入 NCHW float（已归一化），输出 channel-major 的特征 [C*H*W]。
class IBackbone {
 public:
  virtual ~IBackbone() = default;
  virtual std::vector<float> Run(const std::vector<float>& nchw) = 0;
};

#ifdef USE_HISVP
// 需要海思 SVP SDK（svp_nnie 等）。以下为标准 NNIE 前向流程骨架，请按你的 SDK 版本适配：
//   1. SVP_NNIE_LoadModel 从 .wk 加载模型
//   2. 申请输入/输出/任务 MMZ 内存（SVP_NNIE_GetMemSize / 缓存对齐）
//   3. 把归一化后的 NCHW 数据按 NNIE 定点格式填入输入 blob
//   4. SVP_NNIE_Forward 执行前向
//   5. 从输出 blob 读回 float 特征（注意 stride/对齐）
// 由于不同 SVP 版本 API 细节不同，这里只给出结构，具体实现请参考 sample_svp_nnie_software。
#include "svp_nnie_backbone_impl.hpp"  // 用户提供：基于 SVP SDK 的实现
#endif

// ----------- CPU 端图像处理（与 Python/RKNN 版本一致）-----------
BBox ExtendBBox(const BBox& b, float off) {
  return BBox{std::floor(b.x - b.w * off), std::floor(b.y - b.h * off),
              std::floor(b.w * (1 + 2 * off)), std::floor(b.h * (1 + 2 * off))};
}

BBox EnsureBounds(const BBox& b, const cv::Size& s) {
  const float x1 = std::min(std::max(0.f, b.x), (float)s.width);
  const float y1 = std::min(std::max(0.f, b.y), (float)s.height);
  const float x2 = std::min(std::max(0.f, x1 + b.w), (float)s.width);
  const float y2 = std::min(std::max(0.f, y1 + b.h), (float)s.height);
  return BBox{x1, y1, x2 - x1, y2 - y1};
}

BBox ClampBBox(const BBox& in, const cv::Size& s, int min_side = 3) {
  BBox b = EnsureBounds(in, s);
  if (b.w < min_side) { b.w = min_side; b.x -= std::max(0.f, b.x + b.w - s.width); }
  if (b.h < min_side) { b.h = min_side; b.y -= std::max(0.f, b.y + b.h - s.height); }
  return b;
}

// 返回裁剪后(RGB) + 上下文框(context)，用于把预测框映射回原图。
cv::Mat GetExtendedCrop(const cv::Mat& rgb, const BBox& bbox, int crop_size, float offset,
                        cv::Scalar padding, BBox* context_out) {
  BBox c = ExtendBBox(bbox, offset);
  const int cx = (int)c.x, cy = (int)c.y, cw = (int)c.w, ch = (int)c.h;
  const int pl = std::max(-cx, 0), pt = std::max(-cy, 0);
  const int pr = std::max(cx + cw - rgb.cols, 0), pb = std::max(cy + ch - rgb.rows, 0);
  const int rx = cx + pl, ry = cy + pt, rw = cw - pl - pr, rh = ch - pt - pb;
  if (rw <= 0 || rh <= 0) throw std::runtime_error("无效裁剪");
  cv::Mat crop = rgb(cv::Rect(rx, ry, rw, rh));
  cv::Mat padded;
  cv::copyMakeBorder(crop, padded, pt, pb, pl, pr, cv::BORDER_CONSTANT, padding);
  cv::Mat resized;
  cv::resize(padded, resized, cv::Size(crop_size, crop_size), 0, 0, cv::INTER_LINEAR);
  *context_out = c;
  return resized;
}

// RGB -> 归一化 NCHW float
std::vector<float> Preprocess(const cv::Mat& rgb) {
  cv::Mat f;
  rgb.convertTo(f, CV_32FC3, 1.0 / 255.0);
  std::vector<float> nchw(3 * rgb.rows * rgb.cols);
  for (int y = 0; y < rgb.rows; ++y) {
    const cv::Vec3f* row = f.ptr<cv::Vec3f>(y);
    for (int x = 0; x < rgb.cols; ++x)
      for (int ch = 0; ch < 3; ++ch)
        nchw[(ch * rgb.rows + y) * rgb.cols + x] = (row[x][ch] - kMean[ch]) / kStd[ch];
  }
  return nchw;
}

void MakeGrid(std::vector<float>& gx, std::vector<float>& gy) {
  gx.assign(kScoreSize * kScoreSize, 0);
  gy.assign(kScoreSize * kScoreSize, 0);
  const float center = std::floor((float)(kScoreSize / 2));
  for (int r = 0; r < kScoreSize; ++r)
    for (int c = 0; c < kScoreSize; ++c) {
      gx[r * kScoreSize + c] = (c - center) * kTotalStride + kInstanceSize / 2;
      gy[r * kScoreSize + c] = (r - center) * kTotalStride + kInstanceSize / 2;
    }
}

// 解码（smooth=False）：返回 256 裁剪坐标系下的 bbox 与得分。
BBox Decode(const std::vector<float>& bbox_map, const std::vector<float>& cls_logits,
            const std::vector<float>& gx, const std::vector<float>& gy, float* score) {
  int best = 0;
  float best_v = -1e9f;
  for (int i = 0; i < kScoreSize * kScoreSize; ++i) {
    const float s = 1.0f / (1.0f + std::exp(-cls_logits[i]));
    if (s > best_v) { best_v = s; best = i; }
  }
  const int p = best;
  const float x1 = gx[p] - bbox_map[0 * kScoreSize * kScoreSize + p];
  const float y1 = gy[p] - bbox_map[1 * kScoreSize * kScoreSize + p];
  const float w = bbox_map[0 * kScoreSize * kScoreSize + p] + bbox_map[2 * kScoreSize * kScoreSize + p];
  const float h = bbox_map[1 * kScoreSize * kScoreSize + p] + bbox_map[3 * kScoreSize * kScoreSize + p];
  *score = best_v;
  return BBox{x1, y1, w, h};
}

BBox RescaleBBox(const BBox& b, const BBox& ctx) {
  const float ws = ctx.w / (float)kInstanceSize, hs = ctx.h / (float)kInstanceSize;
  return BBox{std::round(b.x * ws + ctx.x), std::round(b.y * hs + ctx.y),
              std::max(3.f, std::round(b.w * ws)), std::max(3.f, std::round(b.h * hs))};
}

struct Args {
  std::string template_wk = "fear_backbone_template.wk";
  std::string search_wk = "fear_backbone_search.wk";
  std::string head_weights = "hi3519_head_weights.bin";
  std::string video = "test.mp4";
  std::string output = "out.mp4";
  std::string bbox = "163,53,45,174";
};

BBox ParseBBox(const std::string& t) {
  std::vector<float> v; std::string tok;
  for (char ch : t) { if (ch == ',' || ch == ' ') { if (!tok.empty()) { v.push_back(std::stof(tok)); tok.clear(); } } else tok.push_back(ch); }
  if (!tok.empty()) v.push_back(std::stof(tok));
  if (v.size() != 4) throw std::runtime_error("--bbox 需要 x,y,w,h 四个数");
  return BBox{v[0], v[1], v[2], v[3]};
}

std::unique_ptr<IBackbone> MakeBackbone(const std::string& wk_path) {
#ifdef USE_HISVP
  return std::make_unique<SvpNnieBackbone>(wk_path);
#else
  (void)wk_path;
  throw std::runtime_error(
      "未编译 NNIE 骨干后端。请在板端用 -DUSE_HISVP=ON 并提供 svp_nnie_backbone_impl.hpp / SVP SDK 重新编译。");
#endif
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args a;
    for (int i = 1; i < argc; i += 2) {
      const std::string k = argv[i];
      if (i + 1 >= argc) throw std::runtime_error("缺少参数值: " + k);
      const std::string v = argv[i + 1];
      if (k == "--template_wk") a.template_wk = v;
      else if (k == "--search_wk") a.search_wk = v;
      else if (k == "--head_weights") a.head_weights = v;
      else if (k == "--video") a.video = v;
      else if (k == "--output") a.output = v;
      else if (k == "--bbox") a.bbox = v;
      else throw std::runtime_error("未知参数: " + k);
    }

    fear::FearHead head(a.head_weights);
    std::unique_ptr<IBackbone> template_bb = MakeBackbone(a.template_wk);
    std::unique_ptr<IBackbone> search_bb = MakeBackbone(a.search_wk);

    std::vector<float> gx, gy;
    MakeGrid(gx, gy);

    cv::VideoCapture cap(a.video);
    if (!cap.isOpened()) throw std::runtime_error("无法打开视频: " + a.video);
    const double fps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 25.0;
    const int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH), H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter writer(a.output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(W, H));

    cv::Mat bgr, rgb;
    if (!cap.read(bgr)) throw std::runtime_error("视频没有帧");
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    BBox bbox = ClampBBox(ParseBBox(a.bbox), rgb.size());
    const cv::Scalar mean_color = cv::mean(rgb);

    // 初始化：模板特征（NNIE 模板骨干，一次）
    BBox ctx;
    cv::Mat tmpl_crop = GetExtendedCrop(rgb, bbox, kTemplateSize, kTemplateBBoxOffset, mean_color, &ctx);
    std::vector<float> template_features = template_bb->Run(Preprocess(tmpl_crop));  // [256*64]

    int frame_idx = 0;
    while (true) {
      cv::rectangle(bgr, cv::Rect((int)bbox.x, (int)bbox.y, (int)bbox.w, (int)bbox.h), cv::Scalar(0, 255, 0), 5);
      writer.write(bgr);
      if (!cap.read(bgr)) break;
      cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

      BBox sctx;
      cv::Mat search_crop = GetExtendedCrop(rgb, bbox, kInstanceSize, kSearchContext, mean_color, &sctx);
      std::vector<float> search_features = search_bb->Run(Preprocess(search_crop));  // [256*256]

      std::vector<float> bbox_map, cls_map;
      head.Forward(search_features, template_features, bbox_map, cls_map);

      float score = 0;
      BBox pred = Decode(bbox_map, cls_map, gx, gy, &score);
      pred = RescaleBBox(pred, sctx);
      bbox = ClampBBox(pred, rgb.size());

      if (++frame_idx % 30 == 0) std::cout << "已处理 " << frame_idx << " 帧, score=" << score << std::endl;
    }
    std::cout << "已写入输出视频: " << a.output << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << std::endl;
    return 1;
  }
}
