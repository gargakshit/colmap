#include "colmap/sensor/ray3d.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(Ray3D, RayBasic) {
  Ray3D ray;
  EXPECT_EQ(ray.ori, Eigen::Vector3d::Zero());
  EXPECT_EQ(ray.dir, Eigen::Vector3d(0.0, 0.0, 1.0));

  Ray3D ray2(Eigen::Vector3d(0.2, 0.1, 0.5), Eigen::Vector3d(1.0, -0.4, 3.0));
  EXPECT_NEAR(ray2.dir.norm(), 1.0, 1e-12);
  EXPECT_EQ(ray2.At(0.0), ray2.ori);
}

TEST(Ray3D, ComputeRefractionPreservesSnellLaw) {
  Eigen::Vector3d normal(0.0, 0.0, 1.0);
  Eigen::Vector3d v = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()) *
                      Eigen::Vector3d::UnitZ();
  const double n1 = 1.0;
  const double n2 = 1.333;
  const double theta1 = std::acos(v.dot(normal));
  ComputeRefraction(normal, n1, n2, &v);
  const double theta2 = std::acos(v.dot(normal));
  EXPECT_NEAR(std::sin(theta1) * n1, std::sin(theta2) * n2, 1e-6);
}

TEST(Ray3D, PlaneIntersection) {
  double d = 0.0;
  const Eigen::Vector3d ray_ori = Eigen::Vector3d::Zero();
  const Eigen::Vector3d ray_dir = Eigen::Vector3d::UnitZ();
  const Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
  EXPECT_TRUE(RayPlaneIntersection(ray_ori, ray_dir, normal, 2.0, &d));
  EXPECT_DOUBLE_EQ(d, 2.0);
}

}  // namespace
}  // namespace colmap
