// Hi3519 CPU 头部段的独立数值自检（仅依赖标准库，可在任意 PC 上编译运行）。
//
// 读取 evaluate/hi3519_dump_head.py 导出的：
//   - hi3519_head_weights.bin  头部权重
//   - hi3519_head_ref.bin      参考输入(search_features/template_features)与 PyTorch 输出(bbox/cls)
// 运行 C++ 头部，与 PyTorch 参考逐元素比较最大绝对误差。
//
// 编译: g++ -O2 -std=c++17 test_fear_head.cpp -o test_fear_head
// 运行: ./test_fear_head /path/to/hi3519_head_weights.bin /path/to/hi3519_head_ref.bin
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fear_head.hpp"

static std::vector<float> ReadN(std::ifstream& f, size_t n) {
  std::vector<float> v(n);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(float)));
  if (!f) throw std::runtime_error("参考文件读取不足");
  return v;
}

static float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

int main(int argc, char** argv) {
  try {
    const std::string weights = argc > 1 ? argv[1] : "outputs/hi3519_split/hi3519_head_weights.bin";
    const std::string ref = argc > 2 ? argv[2] : "outputs/hi3519_split/hi3519_head_ref.bin";

    fear::FearHead head(weights);

    std::ifstream f(ref, std::ios::binary);
    if (!f) throw std::runtime_error("无法打开参考文件: " + ref);
    const std::vector<float> sf = ReadN(f, 256 * 256);
    const std::vector<float> tf = ReadN(f, 256 * 64);
    const std::vector<float> bbox_ref = ReadN(f, 4 * 256);
    const std::vector<float> cls_ref = ReadN(f, 1 * 256);

    std::vector<float> bbox, cls;
    head.Forward(sf, tf, bbox, cls);

    const float bbox_err = MaxAbsDiff(bbox, bbox_ref);
    const float cls_err = MaxAbsDiff(cls, cls_ref);
    std::printf("bbox 最大绝对误差(C++ vs PyTorch) = %.3e\n", bbox_err);
    std::printf("cls  最大绝对误差(C++ vs PyTorch) = %.3e\n", cls_err);

    if (bbox_err < 1e-2f && cls_err < 1e-3f) {
      std::printf("[PASS] C++ 头部与 PyTorch 数值一致\n");
      return 0;
    }
    std::printf("[FAIL] 误差过大\n");
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误: %s\n", e.what());
    return 2;
  }
}
