// Eigen transform-chain benchmark.
//
// Build example:
//   g++ -O3 -march=native -std=c++20 -I /path/to/eigen3 cpp/eigen_transform_chain_bench.cpp -o eigen_transform_chain_bench
//
// Run:
//   ./eigen_transform_chain_bench

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

namespace {

constexpr std::size_t kNumTransforms = 100;
constexpr std::size_t kNumWarmupLoops = 64;
constexpr std::size_t kNumTimedLoops = 4096;
constexpr std::size_t kChainsPerSample = 8;

struct Stats {
  double mean_ns;
  double stddev_ns;
  double max_ns;
};

struct Mat3Vec3Tfm {
  Eigen::Matrix3d rot;
  Eigen::Vector3d xyz;
};

struct QuatVec3Tfm {
  Eigen::Quaterniond rot;
  Eigen::Vector3d xyz;
};

struct So3Vec3Tfm {
  Eigen::Vector3d rot_so3;
  Eigen::Vector3d xyz;
};

struct BenchData {
  std::vector<Eigen::Isometry3d> isometry_chain;
  std::vector<Eigen::Matrix4d> matrix4_chain;
  std::vector<Mat3Vec3Tfm> matrix3_vec3_chain;
  std::vector<QuatVec3Tfm> quat_vec3_chain;
  std::vector<So3Vec3Tfm> so3_vec3_chain;
};

Eigen::Quaterniond so3_exp(const Eigen::Vector3d& so3) {
  const double theta = so3.norm();
  if (theta < 1.0e-12) {
    return Eigen::Quaterniond::Identity();
  }
  const Eigen::Vector3d axis = so3 / theta;
  return Eigen::Quaterniond(Eigen::AngleAxisd(theta, axis));
}

Eigen::Vector3d so3_log(const Eigen::Quaterniond& q_in) {
  Eigen::Quaterniond q = q_in.normalized();
  if (q.w() < 0.0) {
    q.coeffs() = -q.coeffs();
  }
  const Eigen::AngleAxisd aa(q);
  if (std::abs(aa.angle()) < 1.0e-12) {
    return Eigen::Vector3d::Zero();
  }
  return aa.axis() * aa.angle();
}

void compose_assign(Mat3Vec3Tfm* lhs_tfm, const Mat3Vec3Tfm& rhs_tfm) {
  lhs_tfm->xyz = lhs_tfm->rot * rhs_tfm.xyz + lhs_tfm->xyz;
  lhs_tfm->rot = lhs_tfm->rot * rhs_tfm.rot;
}

void compose_assign(QuatVec3Tfm* lhs_tfm, const QuatVec3Tfm& rhs_tfm) {
  lhs_tfm->xyz = lhs_tfm->rot * rhs_tfm.xyz + lhs_tfm->xyz;
  lhs_tfm->rot = lhs_tfm->rot * rhs_tfm.rot;
}

void compose_assign(So3Vec3Tfm* lhs_tfm, const So3Vec3Tfm& rhs_tfm) {
  const Eigen::Quaterniond lhs_q = so3_exp(lhs_tfm->rot_so3);
  const Eigen::Quaterniond rhs_q = so3_exp(rhs_tfm.rot_so3);
  const Eigen::Quaterniond out_q = lhs_q * rhs_q;
  lhs_tfm->xyz = lhs_q * rhs_tfm.xyz + lhs_tfm->xyz;
  lhs_tfm->rot_so3 = so3_log(out_q);
}

Stats compute_stats(const std::vector<double>& samples_ns) {
  const double n = static_cast<double>(samples_ns.size());
  double sum_ns = 0.0;
  for (double sample_ns : samples_ns) {
    sum_ns += sample_ns;
  }
  const double mean_ns = sum_ns / n;

  double var_acc = 0.0;
  for (double sample_ns : samples_ns) {
    const double delta = sample_ns - mean_ns;
    var_acc += delta * delta;
  }
  const double stddev_ns = std::sqrt(var_acc / n);
  const double max_ns = *std::max_element(samples_ns.begin(), samples_ns.end());
  return Stats{mean_ns, stddev_ns, max_ns};
}

BenchData build_transform_chains() {
  BenchData data;
  data.isometry_chain.reserve(kNumTransforms);
  data.matrix4_chain.reserve(kNumTransforms);
  data.matrix3_vec3_chain.reserve(kNumTransforms);
  data.quat_vec3_chain.reserve(kNumTransforms);
  data.so3_vec3_chain.reserve(kNumTransforms);

  for (std::size_t i = 0; i < kNumTransforms; ++i) {
    const double x = static_cast<double>(i);
    const double tx = 0.05 * std::sin(0.31 * x);
    const double ty = 0.04 * std::cos(0.47 * x);
    const double tz = 0.03 * std::sin(0.19 * x);
    Eigen::Vector3d axis_xyz(
        1.0 + 0.11 * std::sin(x), -0.5 + 0.07 * std::cos(x),
        0.9 + 0.05 * std::sin(0.17 * x));
    axis_xyz.normalize();

    const double angle_rad = 0.03 * std::sin(0.13 * x + 0.2);
    const Eigen::Vector3d xyz(tx, ty, tz);

    const Eigen::Quaterniond q(Eigen::AngleAxisd(angle_rad, axis_xyz));
    const Eigen::Matrix3d r = q.toRotationMatrix();

    Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
    iso.linear() = r;
    iso.translation() = xyz;

    data.isometry_chain.push_back(iso);
    data.matrix4_chain.push_back(iso.matrix());
    data.matrix3_vec3_chain.push_back(Mat3Vec3Tfm{r, xyz});
    data.quat_vec3_chain.push_back(QuatVec3Tfm{q, xyz});
    data.so3_vec3_chain.push_back(So3Vec3Tfm{axis_xyz * angle_rad, xyz});
  }

  return data;
}

Stats benchmark_isometry(const std::vector<Eigen::Isometry3d>& chain) {
  std::vector<double> samples_ns;
  samples_ns.reserve(kNumTimedLoops);
  volatile double sink = 0.0;

  for (std::size_t warmup_i = 0; warmup_i < kNumWarmupLoops; ++warmup_i) {
    Eigen::Isometry3d total_tfm = Eigen::Isometry3d::Identity();
    for (const auto& tfm : chain) {
      total_tfm = total_tfm * tfm;
    }
    sink += total_tfm.translation().x();
  }

  for (std::size_t loop_i = 0; loop_i < kNumTimedLoops; ++loop_i) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t sample_i = 0; sample_i < kChainsPerSample; ++sample_i) {
      Eigen::Isometry3d total_tfm = Eigen::Isometry3d::Identity();
      for (const auto& tfm : chain) {
        total_tfm = total_tfm * tfm;
      }
      sink += total_tfm.translation().y();
    }
    const auto stop = std::chrono::steady_clock::now();
    const double dt_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    samples_ns.push_back(dt_ns / static_cast<double>(chain.size() * kChainsPerSample));
  }

  if (sink == std::numeric_limits<double>::infinity()) {
    std::cerr << "unexpected sink value\n";
  }
  return compute_stats(samples_ns);
}

Stats benchmark_matrix4(const std::vector<Eigen::Matrix4d>& chain) {
  std::vector<double> samples_ns;
  samples_ns.reserve(kNumTimedLoops);
  volatile double sink = 0.0;

  for (std::size_t warmup_i = 0; warmup_i < kNumWarmupLoops; ++warmup_i) {
    Eigen::Matrix4d total_tfm = Eigen::Matrix4d::Identity();
    for (const auto& tfm : chain) {
      total_tfm = total_tfm * tfm;
    }
    sink += total_tfm(0, 3);
  }

  for (std::size_t loop_i = 0; loop_i < kNumTimedLoops; ++loop_i) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t sample_i = 0; sample_i < kChainsPerSample; ++sample_i) {
      Eigen::Matrix4d total_tfm = Eigen::Matrix4d::Identity();
      for (const auto& tfm : chain) {
        total_tfm = total_tfm * tfm;
      }
      sink += total_tfm(1, 3);
    }
    const auto stop = std::chrono::steady_clock::now();
    const double dt_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    samples_ns.push_back(dt_ns / static_cast<double>(chain.size() * kChainsPerSample));
  }

  if (sink == std::numeric_limits<double>::infinity()) {
    std::cerr << "unexpected sink value\n";
  }
  return compute_stats(samples_ns);
}

Stats benchmark_matrix3_vec3(const std::vector<Mat3Vec3Tfm>& chain) {
  std::vector<double> samples_ns;
  samples_ns.reserve(kNumTimedLoops);
  volatile double sink = 0.0;

  for (std::size_t warmup_i = 0; warmup_i < kNumWarmupLoops; ++warmup_i) {
    Mat3Vec3Tfm total_tfm{Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()};
    for (const auto& tfm : chain) {
      compose_assign(&total_tfm, tfm);
    }
    sink += total_tfm.xyz.x();
  }

  for (std::size_t loop_i = 0; loop_i < kNumTimedLoops; ++loop_i) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t sample_i = 0; sample_i < kChainsPerSample; ++sample_i) {
      Mat3Vec3Tfm total_tfm{Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()};
      for (const auto& tfm : chain) {
        compose_assign(&total_tfm, tfm);
      }
      sink += total_tfm.xyz.y();
    }
    const auto stop = std::chrono::steady_clock::now();
    const double dt_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    samples_ns.push_back(dt_ns / static_cast<double>(chain.size() * kChainsPerSample));
  }

  if (sink == std::numeric_limits<double>::infinity()) {
    std::cerr << "unexpected sink value\n";
  }
  return compute_stats(samples_ns);
}

Stats benchmark_quat_vec3(const std::vector<QuatVec3Tfm>& chain) {
  std::vector<double> samples_ns;
  samples_ns.reserve(kNumTimedLoops);
  volatile double sink = 0.0;

  for (std::size_t warmup_i = 0; warmup_i < kNumWarmupLoops; ++warmup_i) {
    QuatVec3Tfm total_tfm{Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()};
    for (const auto& tfm : chain) {
      compose_assign(&total_tfm, tfm);
    }
    sink += total_tfm.xyz.x();
  }

  for (std::size_t loop_i = 0; loop_i < kNumTimedLoops; ++loop_i) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t sample_i = 0; sample_i < kChainsPerSample; ++sample_i) {
      QuatVec3Tfm total_tfm{Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()};
      for (const auto& tfm : chain) {
        compose_assign(&total_tfm, tfm);
      }
      sink += total_tfm.xyz.y();
    }
    const auto stop = std::chrono::steady_clock::now();
    const double dt_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    samples_ns.push_back(dt_ns / static_cast<double>(chain.size() * kChainsPerSample));
  }

  if (sink == std::numeric_limits<double>::infinity()) {
    std::cerr << "unexpected sink value\n";
  }
  return compute_stats(samples_ns);
}

Stats benchmark_so3_vec3(const std::vector<So3Vec3Tfm>& chain) {
  std::vector<double> samples_ns;
  samples_ns.reserve(kNumTimedLoops);
  volatile double sink = 0.0;

  for (std::size_t warmup_i = 0; warmup_i < kNumWarmupLoops; ++warmup_i) {
    So3Vec3Tfm total_tfm{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
    for (const auto& tfm : chain) {
      compose_assign(&total_tfm, tfm);
    }
    sink += total_tfm.xyz.x();
  }

  for (std::size_t loop_i = 0; loop_i < kNumTimedLoops; ++loop_i) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t sample_i = 0; sample_i < kChainsPerSample; ++sample_i) {
      So3Vec3Tfm total_tfm{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
      for (const auto& tfm : chain) {
        compose_assign(&total_tfm, tfm);
      }
      sink += total_tfm.xyz.y();
    }
    const auto stop = std::chrono::steady_clock::now();
    const double dt_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    samples_ns.push_back(dt_ns / static_cast<double>(chain.size() * kChainsPerSample));
  }

  if (sink == std::numeric_limits<double>::infinity()) {
    std::cerr << "unexpected sink value\n";
  }
  return compute_stats(samples_ns);
}

void print_stats(const char* label, const Stats& stats) {
  std::cout << label << " per multiply (ns): mean=" << stats.mean_ns
            << ", stddev=" << stats.stddev_ns << ", max=" << stats.max_ns << '\n';
}

}  // namespace

int main() {
  const BenchData data = build_transform_chains();
  const std::size_t samples_per_repr =
      kNumTimedLoops * kNumTransforms * kChainsPerSample;

  const Stats isometry_stats = benchmark_isometry(data.isometry_chain);
  const Stats matrix4_stats = benchmark_matrix4(data.matrix4_chain);
  const Stats matrix3_vec3_stats = benchmark_matrix3_vec3(data.matrix3_vec3_chain);
  const Stats quat_vec3_stats = benchmark_quat_vec3(data.quat_vec3_chain);
  const Stats so3_vec3_stats = benchmark_so3_vec3(data.so3_vec3_chain);

  std::cout << "Eigen transform chain benchmark\n";
  std::cout << "config: transforms=" << kNumTransforms
            << ", warmup_loops=" << kNumWarmupLoops
            << ", timed_loops=" << kNumTimedLoops
            << ", chains_per_sample=" << kChainsPerSample << '\n';
  std::cout << "multiply operations per representation: " << samples_per_repr << '\n';
  print_stats("eigen isometry3", isometry_stats);
  print_stats("eigen matrix4", matrix4_stats);
  print_stats("eigen matrix3+vec3", matrix3_vec3_stats);
  print_stats("eigen quaternion+vec3", quat_vec3_stats);
  print_stats("eigen so3(vec3)+vec3", so3_vec3_stats);

  return 0;
}
