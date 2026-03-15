#include "colmap/estimators/generalized_pose.h"
#include "colmap/estimators/refrac_relative_pose.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/geometry/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/scene/camera.h"

#include <algorithm>
#include <ctime>
#include <fstream>
#include <numeric>

#include <Eigen/LU>
#include <Eigen/SVD>
#include <unsupported/Eigen/KroneckerProduct>

using namespace colmap;

struct PointsData {
  std::vector<Eigen::Vector2d> points2D1;
  std::vector<Eigen::Vector2d> points2D1_refrac;
  std::vector<Eigen::Vector2d> points2D2;
  std::vector<Eigen::Vector2d> points2D2_refrac;
  std::vector<Camera> virtual_cameras1;
  std::vector<Camera> virtual_cameras2;
  std::vector<Rigid3d> virtual_from_reals1;
  std::vector<Rigid3d> virtual_from_reals2;
  Camera best_fit_camera;
  Rigid3d cam2_from_cam1_gt;
};

enum class RelTwoViewMethod {
  kNonRefrac = -1,
  kGR6P = 2,
  kGR6PRefine = 3,
  kBestFit = 4,
  kBestFitRefine = 5,
};

void GenerateRandomSecondViewPose(const Eigen::Vector3d& proj_center,
                                  const double distance,
                                  Rigid3d& cam2_from_cam1) {
  const Eigen::Vector3d target(0.0, 0.0, distance);
  Eigen::Vector3d zaxis = (target - proj_center).normalized();
  Eigen::Vector3d yaxis(0, -1, 0);
  Eigen::Vector3d xaxis = (zaxis.cross(yaxis)).normalized();
  yaxis = (zaxis.cross(xaxis)).normalized();

  Eigen::Matrix3d rotation;
  rotation.col(0) = xaxis;
  rotation.col(1) = yaxis;
  rotation.col(2) = zaxis;
  const Rigid3d cam1_from_cam2(Eigen::Quaterniond(rotation), proj_center);
  cam2_from_cam1 = Inverse(cam1_from_cam2);
}

void GenerateRandom2D2DPoints(const Camera& camera,
                              const size_t num_points,
                              const Rigid3d& cam2_from_cam1_gt,
                              PointsData& points_data,
                              const double noise_level,
                              const double inlier_ratio) {
  points_data.points2D1.clear();
  points_data.points2D1_refrac.clear();
  points_data.points2D2.clear();
  points_data.points2D2_refrac.clear();
  points_data.points2D1.reserve(num_points);
  points_data.points2D1_refrac.reserve(num_points);
  points_data.points2D2.reserve(num_points);
  points_data.points2D2_refrac.reserve(num_points);
  points_data.cam2_from_cam1_gt = cam2_from_cam1_gt;

  const size_t num_inliers =
      static_cast<size_t>(static_cast<double>(num_points) * inlier_ratio);
  size_t cnt = 0;
  while (cnt < num_points) {
    Eigen::Vector2d point2D1_refrac;
    point2D1_refrac.x() = RandomUniformReal(0.5, static_cast<double>(camera.width) - 0.5);
    point2D1_refrac.y() = RandomUniformReal(0.5, static_cast<double>(camera.height) - 0.5);

    const auto ray_refrac = camera.CamFromImgRefrac(point2D1_refrac);
    if (!ray_refrac) {
      continue;
    }

    const double depth = RandomUniformReal(6.0, 8.0);
    const Eigen::Vector3d point3D1 = ray_refrac->At(depth);
    const Eigen::Vector3d point3D2 = cam2_from_cam1_gt * point3D1;

    const auto point2D2_refrac = camera.ImgFromCamRefrac(point3D2);
    const auto point2D1 = camera.ImgFromCam(point3D1);
    const auto point2D2 = camera.ImgFromCam(point3D2);
    if (!point2D2_refrac || !point2D1 || !point2D2) {
      continue;
    }

    if (point2D2_refrac->x() < 0 || point2D2_refrac->x() > camera.width ||
        point2D2_refrac->y() < 0 || point2D2_refrac->y() > camera.height) {
      continue;
    }

    Eigen::Vector2d noisy_point2D1 = *point2D1;
    Eigen::Vector2d noisy_point2D2 = *point2D2;
    Eigen::Vector2d noisy_point2D1_refrac = point2D1_refrac;
    Eigen::Vector2d noisy_point2D2_refrac = *point2D2_refrac;

    const double sigma = cnt < num_inliers ? noise_level : 200.0;
    if (sigma > 0) {
      noisy_point2D1.x() += RandomGaussian(0.0, sigma);
      noisy_point2D1.y() += RandomGaussian(0.0, sigma);
      noisy_point2D1_refrac.x() += RandomGaussian(0.0, sigma);
      noisy_point2D1_refrac.y() += RandomGaussian(0.0, sigma);
      noisy_point2D2.x() += RandomGaussian(0.0, sigma);
      noisy_point2D2.y() += RandomGaussian(0.0, sigma);
      noisy_point2D2_refrac.x() += RandomGaussian(0.0, sigma);
      noisy_point2D2_refrac.y() += RandomGaussian(0.0, sigma);
    }

    points_data.points2D1.push_back(noisy_point2D1);
    points_data.points2D2.push_back(noisy_point2D2);
    points_data.points2D1_refrac.push_back(noisy_point2D1_refrac);
    points_data.points2D2_refrac.push_back(noisy_point2D2_refrac);
    cnt++;
  }

  camera.ComputeVirtuals(points_data.points2D1_refrac,
                         points_data.virtual_cameras1,
                         points_data.virtual_from_reals1);
  camera.ComputeVirtuals(points_data.points2D2_refrac,
                         points_data.virtual_cameras2,
                         points_data.virtual_from_reals2);

  points_data.best_fit_camera =
      BestFitNonRefracCamera(CameraModelId::kOpenCV, camera, 5.0);
}

size_t EstimateRelativePose(Camera& camera,
                            const PointsData& points_data,
                            Rigid3d& cam2_from_cam1,
                            const RelTwoViewMethod method_id) {
  const size_t num_points = points_data.points2D1.size();
  TwoViewGeometryOptions options;
  options.compute_relative_pose = true;
  options.ransac_options.max_error = 4.0;

  FeatureMatches matches;
  matches.reserve(num_points);
  for (size_t i = 0; i < num_points; i++) {
    matches.emplace_back(i, i);
  }

  TwoViewGeometry two_view_geometry;
  switch (method_id) {
    case RelTwoViewMethod::kNonRefrac:
      two_view_geometry = EstimateCalibratedTwoViewGeometry(
          camera, points_data.points2D1, camera, points_data.points2D2, matches, options);
      break;
    case RelTwoViewMethod::kGR6P:
      options.enable_refraction = true;
      two_view_geometry = EstimateCalibratedTwoViewGeometry(camera,
                                                            points_data.points2D1_refrac,
                                                            camera,
                                                            points_data.points2D2_refrac,
                                                            matches,
                                                            options);
      break;
    case RelTwoViewMethod::kGR6PRefine:
      options.enable_refraction = true;
      two_view_geometry = EstimateCalibratedTwoViewGeometry(camera,
                                                            points_data.points2D1_refrac,
                                                            camera,
                                                            points_data.points2D2_refrac,
                                                            matches,
                                                            options);
      break;
    case RelTwoViewMethod::kBestFit:
      two_view_geometry = EstimateRefractiveTwoViewGeometryUseBestFit(
          points_data.best_fit_camera,
          points_data.points2D1_refrac,
          points_data.virtual_cameras1,
          points_data.virtual_from_reals1,
          points_data.best_fit_camera,
          points_data.points2D2_refrac,
          points_data.virtual_cameras2,
          points_data.virtual_from_reals2,
          matches,
          options,
          false);
      break;
    case RelTwoViewMethod::kBestFitRefine:
      two_view_geometry = EstimateRefractiveTwoViewGeometryUseBestFit(
          points_data.best_fit_camera,
          points_data.points2D1_refrac,
          points_data.virtual_cameras1,
          points_data.virtual_from_reals1,
          points_data.best_fit_camera,
          points_data.points2D2_refrac,
          points_data.virtual_cameras2,
          points_data.virtual_from_reals2,
          matches,
          options,
          true);
      break;
    default:
      LOG(ERROR) << "Relative two-view method does not exist!";
      return 0;
  }

  if (!two_view_geometry.cam2_from_cam1.has_value()) {
    return 0;
  }
  cam2_from_cam1 = *two_view_geometry.cam2_from_cam1;
  if (two_view_geometry.inlier_matches.empty() || two_view_geometry.tri_angle == 0.0) {
    return 0;
  }
  return two_view_geometry.inlier_matches.size();
}

void RelativePoseError(const Rigid3d& cam2_from_cam1_gt,
                       const Rigid3d& cam2_from_cam1_est,
                       double& rotation_error,
                       double& angular_error) {
  Rigid3d gt = cam2_from_cam1_gt;
  gt.translation().normalize();
  Rigid3d est = cam2_from_cam1_est;
  est.translation().normalize();

  const Eigen::Quaterniond rotation_diff =
      cam2_from_cam1_gt.rotation() * cam2_from_cam1_est.rotation().inverse();
  rotation_error = RadToDeg(Eigen::AngleAxisd(rotation_diff).angle());

  double cos_theta = gt.translation().dot(est.translation());
  if (cos_theta < 0) {
    cos_theta = -cos_theta;
  }
  cos_theta = std::min(1.0, cos_theta);
  angular_error = RadToDeg(std::acos(cos_theta));
}

void Evaluate(Camera& camera,
              const size_t num_points,
              const size_t num_exps,
              const double inlier_ratio,
              const bool is_flatport,
              const std::vector<RelTwoViewMethod>& methods,
              const std::string& output_path) {
  const std::vector<double> noise_levels = {0.0, 0.2, 0.5, 0.8, 1.2, 1.5, 1.8, 2.0};
  std::ofstream file(output_path, std::ios::out);

  for (const double noise : noise_levels) {
    std::vector<PointsData> datasets;
    datasets.reserve(num_exps);

    for (size_t i = 0; i < num_exps; i++) {
      if (is_flatport) {
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
      } else {
        camera.refrac_model_id = CameraRefracModelId::kDomePort;
        Eigen::Vector3d decentering(RandomUniformReal(-0.01, 0.01),
                                    RandomUniformReal(-0.01, 0.01),
                                    RandomUniformReal(-0.03, 0.03));
        camera.refrac_params = {decentering[0],
                                decentering[1],
                                decentering[2],
                                RandomUniformReal(0.05, 0.07),
                                RandomUniformReal(0.005, 0.02),
                                1.0,
                                1.52,
                                1.334};
      }

      const Eigen::Vector3d proj_center(RandomUniformReal(8.0, 10.0),
                                        RandomUniformReal(-2.5, 2.5),
                                        RandomUniformReal(-2.5, 2.5));
      const double distance = RandomUniformReal(6.0, 8.0);
      Rigid3d cam2_from_cam1;
      GenerateRandomSecondViewPose(proj_center, distance, cam2_from_cam1);

      PointsData points_data;
      GenerateRandom2D2DPoints(
          camera, num_points, cam2_from_cam1, points_data, noise, inlier_ratio);
      datasets.push_back(points_data);
    }

    file << noise;
    for (const RelTwoViewMethod method_id : methods) {
      std::vector<double> rotation_errors;
      std::vector<double> angular_errors;
      std::vector<double> inlier_ratios;

      for (size_t i = 0; i < num_exps; i++) {
        const PointsData& points_data = datasets[i];
        Rigid3d cam2_from_cam1_est;
        const size_t num_inliers =
            EstimateRelativePose(camera, points_data, cam2_from_cam1_est, method_id);
        if (num_inliers == 0) {
          continue;
        }
        double rotation_error = 0.0;
        double angular_error = 0.0;
        RelativePoseError(points_data.cam2_from_cam1_gt,
                          cam2_from_cam1_est,
                          rotation_error,
                          angular_error);
        rotation_errors.push_back(rotation_error);
        angular_errors.push_back(angular_error);
        inlier_ratios.push_back(static_cast<double>(num_inliers) /
                                static_cast<double>(num_points));
      }

      file << " " << Mean(rotation_errors) << " " << StdDev(rotation_errors) << " "
           << Mean(angular_errors) << " " << StdDev(angular_errors) << " "
           << Mean(inlier_ratios);
    }
    file << std::endl;
  }
}

int main(int argc, char* argv[]) {
  SetPRNGSeed(time(nullptr));

  Camera camera;
  camera.width = 1920;
  camera.height = 1080;
  camera.model_id = CameraModelId::kPinhole;
  camera.params = {1297.3655404279762, 1297.3655404279762, 960.0, 540.0};

  const std::vector<RelTwoViewMethod> methods = {RelTwoViewMethod::kNonRefrac,
                                                 RelTwoViewMethod::kGR6P,
                                                 RelTwoViewMethod::kBestFit};
  Evaluate(camera,
           200,
           200,
           0.7,
           false,
           methods,
           "eval_refrac_rel_pose_dome_num_points_200_inlier_ratio_0.7.txt");
  return EXIT_SUCCESS;
}
