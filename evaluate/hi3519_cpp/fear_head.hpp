// FEAR 跟踪器 CPU 头部段（BoxTower）的 C++ 实现，用于 Hi3519 板端。
//
// NNIE 跑不了头部的互相关 MatMul / Exp，故头部在 ARM 上用本文件实现。
// 权重由 evaluate/hi3519_dump_head.py 导出（已折叠 BatchNorm），按固定顺序读取。
//
// 输入：
//   search_features  [256, 16, 16]（channel-major，来自 NNIE 搜索骨干）
//   template_features[256, 8, 8]  （channel-major，来自 NNIE 模板骨干，初始化时算一次）
// 输出：
//   bbox [4, 16, 16]（已 exp）、cls [1, 16, 16]（logits，未 sigmoid）
//
// 仅依赖 C++ 标准库，可独立编译自检（见 test_fear_head.cpp）。
#ifndef FEAR_HI3519_HEAD_HPP
#define FEAR_HI3519_HEAD_HPP

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fear {

constexpr int kSearchHW = 16 * 16;   // 搜索特征空间大小 P
constexpr int kTemplateHW = 8 * 8;   // 模板特征空间大小 K
constexpr int kFeatC = 256;          // 骨干输出通道数
constexpr int kCorrC = 64;           // 互相关产生的通道数(=模板空间 K)

// 一个「depthwise(3x3,pad1) + pointwise(1x1) + 可选 ReLU」卷积块（BN 已折叠进 pointwise）。
struct ConvBlock {
  int C = 0;       // depthwise 通道数(=输入通道数)
  int Cout = 0;    // pointwise 输出通道数
  int Cin = 0;     // pointwise 输入通道数(==C)
  bool relu = true;
  std::vector<float> dw_w;  // [C,3,3]
  std::vector<float> dw_b;  // [C]
  std::vector<float> pw_w;  // [Cout,Cin]
  std::vector<float> pw_b;  // [Cout]

  // in: [C, HW] channel-major -> out: [Cout, HW]
  std::vector<float> apply(const std::vector<float>& in, int hw, int side) const {
    std::vector<float> dw(static_cast<size_t>(C) * hw, 0.0f);
    for (int c = 0; c < C; ++c) {
      const float* wk = &dw_w[static_cast<size_t>(c) * 9];
      const float* src = &in[static_cast<size_t>(c) * hw];
      float* dst = &dw[static_cast<size_t>(c) * hw];
      for (int row = 0; row < side; ++row) {
        for (int col = 0; col < side; ++col) {
          float acc = dw_b[c];
          for (int ky = 0; ky < 3; ++ky) {
            const int ir = row + ky - 1;
            if (ir < 0 || ir >= side) continue;
            for (int kx = 0; kx < 3; ++kx) {
              const int ic = col + kx - 1;
              if (ic < 0 || ic >= side) continue;
              acc += wk[ky * 3 + kx] * src[ir * side + ic];
            }
          }
          dst[row * side + col] = acc;
        }
      }
    }
    std::vector<float> out(static_cast<size_t>(Cout) * hw, 0.0f);
    for (int co = 0; co < Cout; ++co) {
      const float* w = &pw_w[static_cast<size_t>(co) * Cin];
      const float bias = pw_b[co];
      float* dst = &out[static_cast<size_t>(co) * hw];
      for (int p = 0; p < hw; ++p) {
        float acc = bias;
        for (int ci = 0; ci < Cin; ++ci) {
          acc += w[ci] * dw[static_cast<size_t>(ci) * hw + p];
        }
        dst[p] = relu ? (acc > 0.0f ? acc : 0.0f) : acc;
      }
    }
    return out;
  }
};

class FearHead {
 public:
  explicit FearHead(const std::string& weights_path) { Load(weights_path); }

  // search_features: [256*256], template_features: [256*64]
  // 输出 bbox[4*256]（已 exp）与 cls[1*256]（logits）
  void Forward(const std::vector<float>& sf, const std::vector<float>& tf,
               std::vector<float>& bbox_out, std::vector<float>& cls_out) const {
    const std::vector<float> cls_x = cls_encode_.apply(sf, kSearchHW, 16);
    const std::vector<float> reg_x = reg_encode_.apply(sf, kSearchHW, 16);

    const std::vector<float> cls_cat = Correlate(tf, cls_x);
    const std::vector<float> reg_cat = Correlate(tf, reg_x);

    const std::vector<float> cls_dw = cls_dw_enc_.apply(cls_cat, kSearchHW, 16);
    const std::vector<float> reg_dw = reg_dw_enc_.apply(reg_cat, kSearchHW, 16);

    std::vector<float> x_reg = bbox_tower0_.apply(reg_dw, kSearchHW, 16);
    x_reg = bbox_tower1_.apply(x_reg, kSearchHW, 16);
    std::vector<float> bbox = bbox_pred_.apply(x_reg, kSearchHW, 16);  // [4,256]
    for (int co = 0; co < 4; ++co) {
      for (int p = 0; p < kSearchHW; ++p) {
        const size_t idx = static_cast<size_t>(co) * kSearchHW + p;
        bbox[idx] = std::exp(adjust_ * bbox[idx] + bias_[co]);
      }
    }

    std::vector<float> c = cls_tower0_.apply(cls_dw, kSearchHW, 16);
    c = cls_tower1_.apply(c, kSearchHW, 16);
    std::vector<float> cls = cls_pred_.apply(c, kSearchHW, 16);  // [1,256]
    for (int p = 0; p < kSearchHW; ++p) cls[p] *= 0.1f;

    bbox_out = std::move(bbox);
    cls_out = std::move(cls);
  }

 private:
  // s[k,p] = sum_c tf[c,k] * x[c,p]; 然后 cat([x(256), s(64)]) -> [320, 256]
  static std::vector<float> Correlate(const std::vector<float>& tf, const std::vector<float>& x) {
    std::vector<float> cat(static_cast<size_t>(kFeatC + kCorrC) * kSearchHW, 0.0f);
    // 前 256 通道直接拷贝 x
    std::copy(x.begin(), x.end(), cat.begin());
    // 后 64 通道为互相关结果
    for (int k = 0; k < kCorrC; ++k) {
      float* dst = &cat[static_cast<size_t>(kFeatC + k) * kSearchHW];
      for (int p = 0; p < kSearchHW; ++p) {
        float acc = 0.0f;
        for (int c = 0; c < kFeatC; ++c) {
          acc += tf[static_cast<size_t>(c) * kTemplateHW + k] * x[static_cast<size_t>(c) * kSearchHW + p];
        }
        dst[p] = acc;
      }
    }
    return cat;
  }

  static void ReadFloats(std::ifstream& f, std::vector<float>& dst, size_t n) {
    dst.resize(n);
    f.read(reinterpret_cast<char*>(dst.data()), static_cast<std::streamsize>(n * sizeof(float)));
    if (!f) throw std::runtime_error("权重文件读取不足，可能与导出顺序/版本不匹配");
  }

  static ConvBlock ReadBlock(std::ifstream& f, int C, int Cout, int Cin, bool relu) {
    ConvBlock b;
    b.C = C; b.Cout = Cout; b.Cin = Cin; b.relu = relu;
    ReadFloats(f, b.dw_w, static_cast<size_t>(C) * 9);
    ReadFloats(f, b.dw_b, static_cast<size_t>(C));
    ReadFloats(f, b.pw_w, static_cast<size_t>(Cout) * Cin);
    ReadFloats(f, b.pw_b, static_cast<size_t>(Cout));
    return b;
  }

  void Load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("无法打开头部权重文件: " + path);
    cls_encode_ = ReadBlock(f, 256, 256, 256, true);
    reg_encode_ = ReadBlock(f, 256, 256, 256, true);
    cls_dw_enc_ = ReadBlock(f, 320, 256, 320, true);
    reg_dw_enc_ = ReadBlock(f, 320, 256, 320, true);
    bbox_tower0_ = ReadBlock(f, 256, 256, 256, true);
    bbox_tower1_ = ReadBlock(f, 256, 256, 256, true);
    cls_tower0_ = ReadBlock(f, 256, 256, 256, true);
    cls_tower1_ = ReadBlock(f, 256, 256, 256, true);
    bbox_pred_ = ReadBlock(f, 256, 4, 256, false);
    cls_pred_ = ReadBlock(f, 256, 1, 256, false);
    std::vector<float> adjust, bias;
    ReadFloats(f, adjust, 1);
    ReadFloats(f, bias, 4);
    adjust_ = adjust[0];
    bias_ = bias;
  }

  ConvBlock cls_encode_, reg_encode_, cls_dw_enc_, reg_dw_enc_;
  ConvBlock bbox_tower0_, bbox_tower1_, cls_tower0_, cls_tower1_;
  ConvBlock bbox_pred_, cls_pred_;
  float adjust_ = 0.1f;
  std::vector<float> bias_;
};

}  // namespace fear

#endif  // FEAR_HI3519_HEAD_HPP
