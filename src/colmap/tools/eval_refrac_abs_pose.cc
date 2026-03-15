#include "colmap/estimators/generalized_pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/scene/camera.h"

#include <ctime>
#include <fstream>
#include <numeric>

using namespace colmap;

struct PointsData {
  std::vector<Eigen::Vector2d> points2D;
  std::vector<Eigen::Vector2d> points2D_refrac;
  std::vector<Eigen::Vector3d> points3D;
  std::vector<Camera> virtual_cameras;
  std::vector<Rigid3d> virtual_from_reals;
  Rigid3d cam_from_world_gt;
};

void GenerateRandom3D2DPoints(const Camera& camera,
                              const size_t num_points,
                              const Rigid3d& cam_from_world_gt,
                              PointsData& points_data,
                              const double noise_level,
                              const double inlier_ratio) {
  points_data.points2D.clear();
  points_data.points2D_refrac.clear();
  points_data.points3D.clear();
  points_data.points2D.reserve(num_points);
  points_data.points2D_refrac.reserve(num_points);
  points_data.points3D.reserve(num_points);
  points_data.cam_from_world_gt = cam_from_world_gt;

  const size_t num_inliers =
      static_cast<size_t>(static_cast<double>(num_points) * inlier_ratio);
  size_t cnt = 0;
  for (size_t i = 0; i < num_points; i++) {
    Eigen::Vector2d point2D_refrac;
    point2D_refrac.x() = RandomUniformReal(0.5, static_cast<double>(camera.width) - 0.5);
    point2D_refrac.y() = RandomUniformReal(0.5, static_cast<double>(camera.height) - 0.5);

    const double depth = RandomUniformReal(0.5, 10.0);
    const auto point3D_local = camera.CamFromImgRefracPoint(point2D_refrac, depth);
    if (!point3D_local) {
      continue;
    }

    const Eigen::Vector3d point3D_world = Inverse(cam_from_world_gt) * *point3D_local;
    const auto point2D = camera.ImgFromCam(*point3D_local);
    if (!point2D) {
      continue;
    }

    Eigen::Vector2d noisy_point2D_refrac = point2D_refrac;
    Eigen::Vector2d noisy_point2D = *point2D;
    if (cnt < num_inliers) {
      noisy_point2D_refrac.x() += RandomGaussian(0.0, noise_level);
      noisy_point2D_refrac.y() += RandomGaussian(0.0, noise_level);
      noisy_point2D.x() += RandomGaussian(0.0, noise_level);
      noisy_point2D.y() += RandomGaussian(0.0, noise_level);
    } else {
      noisy_point2D_refrac.x() += RandomGaussian(0.0, 200.0);
      noisy_point2D_refrac.y() += RandomGaussian(0.0, 200.0);
      noisy_point2D.x() += RandomGaussian(0.0, 200.0);
      noisy_point2D.y() += RandomGaussian(0.0, 200.0);
    }

    points_data.points2D.push_back(noisy_point2D);
    points_data.points2D_refrac.push_back(noisy_point2D_refrac);
    points_data.points3D.push_back(point3D_world);
    cnt++;
  }

  camera.ComputeVirtuals(points_data.points2D_refrac,
                         points_data.virtual_cameras,
                         points_data.virtual_from_reals);
}

size_t EstimatePose(Camera& camera,
                    const PointsData& points_data,
                    Rigid3d& cam_from_world,
                    const bool is_refractive) {
  AbsolutePoseEstimationOptions abs_pose_options;
  abs_pose_options.ransac_options.max_error = 12.0;
  abs_pose_options.ransac_options.min_inlier_ratio = 0.25;
  abs_pose_options.ransac_options.min_num_trials = 100;
  abs_pose_options.ransac_options.max_num_trials = 10000;
  abs_pose_options.ransac_options.confidence = 0.99999;

  size_t num_inliers = 0;
  std::vector<char> inlier_mask;

  if (!is_refractive) {
    if (!EstimateAbsolutePose(abs_pose_options,
                              points_data.points2D,
                              points_data.points3D,
                              &cam_from_world,
                              &camera,
                              &num_inliers,
                              &inlier_mask)) {
      LOG(ERROR) << "ERROR: Pose estimation failed";
      return 0;
    }
  } else {
    std::vector<size_t> camera_idxs(points_data.points2D_refrac.size());
    std::iota(camera_idxs.begin(), camera_idxs.end(), 0);
    if (!EstimateGeneralizedAbsolutePose(abs_pose_options.ransac_options,
                                         points_data.points2D_refrac,
                                         points_data.points3D,
                                         camera_idxs,
                                         points_data.virtual_from_reals,
                                         points_data.virtual_cameras,
                                         &cam_from_world,
                                         &num_inliers,
                                         &inlier_mask)) {
      LOG(ERROR) << "ERROR: Pose estimation failed";
      return 0;
    }
  }

  if (num_inliers < 6) {
    LOG(ERROR) << "ERROR: Pose estimation failed, insufficient number of inliers";
    return 0;
  }
  return num_inliers;
}

void PoseError(const Rigid3d& cam1_from_world,
               const Rigid3d& cam2_from_world,
               double& rotation_error,
               double& position_error) {
  const Rigid3d cam2_from_cam1 = cam2_from_world * Inverse(cam1_from_world);
  rotation_error =
      RadToDeg(Eigen::AngleAxisd(cam2_from_cam1.rotation()).angle());
  position_error =
      (Inverse(cam1_from_world).translation() -
       Inverse(cam2_from_world).translation())
          .norm() *
      1000.0;
}

void Evaluate(Camera& camera,
              const size_t num_points,
              const size_t num_exps,
              const double inlier_ratio,
              const bool is_flatport,
              const std::string& output_path) {
  const std::vector<double> noise_levels = {0.0, 0.2, 0.5, 0.8, 1.2, 1.5, 1.8, 2.0};

  std::ofstream file(output_path, std::ios::out);
  file << "# noise_level rot_error_mean rot_error_std pos_error_mean "
          "pos_error_std rot_error_refrac_mean rot_error_refrac_std "
          "pos_error_refrac_mean pos_error_refrac_std"
       << std::endl;

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

      const double qx = RandomUniformReal(0.0, 1.0);
      const double tx = RandomUniformReal(0.0, 1.0);
      const Rigid3d cam_from_world(
          Eigen::Quaterniond(1.0, qx, 0.0, 0.0).normalized(),
          Eigen::Vector3d(tx, 0.0, 0.0));

      PointsData points_data;
      GenerateRandom3D2DPoints(
          camera, num_points, cam_from_world, points_data, noise, inlier_ratio);
      datasets.push_back(points_data);
    }

    std::vector<double> rotation_errors;
    std::vector<double> position_errors;
    std::vector<double> rotation_errors_refrac;
    std::vector<double> position_errors_refrac;
    std::vector<double> inlier_ratios;
    std::vector<double> inlier_ratios_refrac;

    for (size_t i = 0; i < num_exps; i++) {
      const PointsData& points_data = datasets[i];
      Rigid3d cam_from_world_est;
      const size_t num_inliers =
          EstimatePose(camera, points_data, cam_from_world_est, false);
      double rotation_error = 0.0;
      double position_error = 0.0;
      PoseError(points_data.cam_from_world_gt,
                cam_from_world_est,
                rotation_error,
                position_error);
      rotation_errors.push_back(rotation_error);
      position_errors.push_back(position_error);
      inlier_ratios.push_back(static_cast<double>(num_inliers) /
                              static_cast<double>(num_points));
    }

    for (size_t i = 0; i < num_exps; i++) {
      const PointsData& points_data = datasets[i];
      Rigid3d cam_from_world_est_refrac;
      const size_t num_inliers =
          EstimatePose(camera, points_data, cam_from_world_est_refrac, true);
      double rotation_error_refrac = 0.0;
      double position_error_refrac = 0.0;
      PoseError(points_data.cam_from_world_gt,
                cam_from_world_est_refrac,
                rotation_error_refrac,
                position_error_refrac);
      rotation_errors_refrac.push_back(rotation_error_refrac);
      position_errors_refrac.push_back(position_error_refrac);
      inlier_ratios_refrac.push_back(static_cast<double>(num_inliers) /
                                     static_cast<double>(num_points));
    }

    file << noise << " " << Mean(rotation_errors) << " " << StdDev(rotation_errors)
         << " " << Mean(position_errors) << " " << StdDev(position_errors) << " "
         << Mean(rotation_errors_refrac) << " " << StdDev(rotation_errors_refrac)
         << " " << Mean(position_errors_refrac) << " "
         << StdDev(position_errors_refrac) << " " << Mean(inlier_ratios) << " "
         << Mean(inlier_ratios_refrac) << std::endl;
  }
}

int main(int argc, char* argv[]) {
  SetPRNGSeed(time(nullptr));

  Camera camera;
  camera.width = 1920;
  camera.height = 1080;
  camera.model_id = CameraModelId::kPinhole;
  camera.params = {1297.3655404279762, 1297.3655404279762, 960.0, 540.0};

  const size_t num_points = 200;
  const double inlier_ratio = 0.7;
  const bool is_flatport = false;
  const std::string output_path =
      "eval_refrac_abs_pose_dome_num_points_200_inlier_ratio_0.7.txt";

  Evaluate(camera, num_points, 1000, inlier_ratio, is_flatport, output_path);
  return EXIT_SUCCESS;
}
