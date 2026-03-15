#include "colmap/sensor/models_refrac.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CameraRefracModel, Metadata) {
  EXPECT_TRUE(ExistsCameraRefracModelWithId(CameraRefracModelId::kFlatPort));
  EXPECT_EQ(CameraRefracModelNameToId("DOMEPORT"),
            CameraRefracModelId::kDomePort);
  EXPECT_EQ(CameraRefracModelIdToName(CameraRefracModelId::kFlatPort),
            "FLATPORT");
}

TEST(CameraRefracModel, FlatPortRoundTrip) {
  const std::vector<double> cam_params = {1200.0, 640.0, 480.0};
  const std::vector<double> refrac_params = {0.0, 0.0, 1.0, 0.05,
                                             0.007, 1.0, 1.49, 1.333};
  const Eigen::Vector3d cam_point(0.1, -0.05, 2.0);

  const auto xy = CameraRefracModelImgFromCam(CameraModelId::kSimplePinhole,
                                              CameraRefracModelId::kFlatPort,
                                              cam_params,
                                              refrac_params,
                                              cam_point);
  ASSERT_TRUE(xy.has_value());

  const auto uvw =
      CameraRefracModelCamFromImgPoint(CameraModelId::kSimplePinhole,
                                       CameraRefracModelId::kFlatPort,
                                       cam_params,
                                       refrac_params,
                                       *xy,
                                       cam_point.norm());
  ASSERT_TRUE(uvw.has_value());
  EXPECT_NEAR(uvw->x(), cam_point.x(), 1e-5);
  EXPECT_NEAR(uvw->y(), cam_point.y(), 1e-5);
  EXPECT_NEAR(uvw->z(), cam_point.z(), 1e-5);
}

TEST(CameraRefracModel, DomePortBackprojection) {
  const std::vector<double> cam_params = {1000.0, 1000.0, 640.0, 480.0};
  const std::vector<double> refrac_params = {0.0, 0.0, 0.0, 0.1,
                                             0.005, 1.0, 1.49, 1.333};
  const auto ray = CameraRefracModelCamFromImg(CameraModelId::kPinhole,
                                               CameraRefracModelId::kDomePort,
                                               cam_params,
                                               refrac_params,
                                               Eigen::Vector2d(640.0, 480.0));
  ASSERT_TRUE(ray.has_value());
  EXPECT_NEAR(ray->dir.norm(), 1.0, 1e-12);
}

}  // namespace
}  // namespace colmap
