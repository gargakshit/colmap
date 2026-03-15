#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/util/eigen_alignment.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

class RefracRelPoseEstimator {
 public:
  struct X_t {
    Rigid3d virtual_from_real;
    Eigen::Vector3d ray_in_virtual;
  };

  using Y_t = X_t;
  using M_t = Rigid3d;

  static const int kMinNumSamples = 17;

  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       std::vector<M_t>* models);

  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& cam2_from_cam1,
                        std::vector<double>* residuals);
};

}  // namespace colmap
