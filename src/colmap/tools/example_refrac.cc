#include "colmap/math/random.h"
#include "colmap/scene/camera.h"
#include "colmap/util/logging.h"

#include <iostream>

using namespace colmap;

int main(int argc, char** argv) {
  Camera camera;
  camera.width = 1920;
  camera.height = 1080;
  camera.model_id = CameraModelId::kSimpleRadial;
  camera.params = {
      1297.3655404279762, 1297.3655404279762, 960.0, 540.0, 0.01};

  camera.refrac_model_id = CameraRefracModelId::kFlatPort;
  Eigen::Vector3d int_normal(RandomUniformReal(-0.2, 0.2),
                             RandomUniformReal(-0.2, 0.2),
                             RandomUniformReal(0.8, 1.2));
  int_normal.normalize();
  camera.refrac_params = {int_normal[0],
                          int_normal[1],
                          int_normal[2],
                          RandomUniformReal(0.001, 0.05),
                          RandomUniformReal(0.002, 0.2),
                          1.0,
                          1.52,
                          1.334};

  const Eigen::Vector2d point2D(
      RandomUniformReal(0.5, static_cast<double>(camera.width - 0.5)),
      RandomUniformReal(0.5, static_cast<double>(camera.height - 0.5)));

  const auto ray_w = camera.CamFromImgRefrac(point2D);
  if (!ray_w) {
    LOG(ERROR) << "Failed to compute refracted ray";
    return EXIT_FAILURE;
  }

  LOG(INFO) << "ray origin on the interface: " << ray_w->ori.transpose()
            << " , ray direction in the water: " << ray_w->dir.transpose();

  const double depth = RandomUniformReal(0.5, 10.0);
  const Eigen::Vector3d point3D = ray_w->At(depth);

  const auto projection_refrac = camera.ImgFromCamRefrac(point3D);
  if (!projection_refrac) {
    LOG(ERROR) << "Failed to project refractive point";
    return EXIT_FAILURE;
  }

  LOG(INFO) << " Original 2D point: " << point2D.transpose();
  LOG(INFO) << "Projected 2D point: " << projection_refrac->transpose();
  return EXIT_SUCCESS;
}

