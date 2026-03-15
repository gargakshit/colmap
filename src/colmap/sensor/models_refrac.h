#pragma once

#include "colmap/sensor/models.h"
#include "colmap/sensor/ray3d.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

namespace colmap {

enum class CameraRefracModelId {
  kInvalid = -1,
  kFlatPort = 0,
  kDomePort = 1,
};

#ifndef CAMERA_REFRAC_MODEL_DEFINITIONS
#define CAMERA_REFRAC_MODEL_DEFINITIONS(                                       \
    refrac_model_id_val, refrac_model_name_val, num_params_val)                \
  static constexpr size_t num_params = num_params_val;                         \
  static constexpr CameraRefracModelId refrac_model_id = refrac_model_id_val;  \
  static const std::string refrac_model_name;                                  \
  static const std::string refrac_params_info;                                 \
  static const std::vector<size_t> optimizable_params_idxs;                    \
                                                                               \
  static inline CameraRefracModelId InitializeRefracModelId() {                \
    return refrac_model_id_val;                                                \
  }                                                                            \
  static inline std::string InitializeRefracModelName() {                      \
    return refrac_model_name_val;                                              \
  }                                                                            \
  static inline std::string InitializeRefracParamsInfo();                      \
  static inline std::vector<size_t> InitializeOptimizableParamsIdxs();         \
                                                                               \
  template <typename CameraModel, typename T>                                  \
  static bool ImgFromCam(                                                      \
      const T* cam_params, const T* refrac_params, T u, T v, T w, T* x, T* y); \
                                                                               \
  template <typename CameraModel, typename T>                                  \
  static bool CamFromImg(const T* cam_params,                                  \
                         const T* refrac_params,                               \
                         T x,                                                  \
                         T y,                                                  \
                         Eigen::Matrix<T, 3, 1>* ori,                          \
                         Eigen::Matrix<T, 3, 1>* dir);                         \
  template <typename CameraModel, typename T>                                  \
  static bool CamFromImgPoint(const T* cam_params,                             \
                              const T* refrac_params,                          \
                              T x,                                             \
                              T y,                                             \
                              T d,                                             \
                              Eigen::Matrix<T, 3, 1>* uvw);                    \
  template <typename T>                                                        \
  static void RefractionAxis(const T* refrac_params,                           \
                             Eigen::Matrix<T, 3, 1>* refraction_axis);
#endif

#ifndef CAMERA_REFRAC_MODEL_CASES
#define CAMERA_REFRAC_MODEL_CASES    \
  CAMERA_REFRAC_MODEL_CASE(FlatPort) \
  CAMERA_REFRAC_MODEL_CASE(DomePort)
#endif

#ifndef CAMERA_COMBINATION_MODEL_CASES
#define CAMERA_COMBINATION_MODEL_CASES                                     \
  CAMERA_COMBINATION_MODEL_CASE(FlatPort, SimplePinholeCameraModel)        \
  CAMERA_COMBINATION_MODEL_CASE(FlatPort, PinholeCameraModel)              \
  CAMERA_COMBINATION_MODEL_CASE(FlatPort, SimpleRadialCameraModel)         \
  CAMERA_COMBINATION_MODEL_CASE(FlatPort, SimpleRadialFisheyeCameraModel)  \
  CAMERA_COMBINATION_MODEL_CASE(FlatPort, RadialCameraModel)               \
  CAMERA_COMBINATION_MODEL_CASE(FlatPort, RadialFisheyeCameraModel)        \
  CAMERA_COMBINATION_MODEL_CASE(FlatPort, OpenCVCameraModel)               \
  CAMERA_COMBINATION_MODEL_CASE(FlatPort, OpenCVFisheyeCameraModel)        \
  CAMERA_COMBINATION_MODEL_CASE(FlatPort, FullOpenCVCameraModel)           \
  CAMERA_COMBINATION_MODEL_CASE(FlatPort, FOVCameraModel)                  \
  CAMERA_COMBINATION_MODEL_CASE(FlatPort, ThinPrismFisheyeCameraModel)     \
  CAMERA_COMBINATION_MODEL_CASE(FlatPort, RadTanThinPrismFisheyeModel)     \
  CAMERA_COMBINATION_MODEL_CASE(FlatPort, SimpleDivisionCameraModel)       \
  CAMERA_COMBINATION_MODEL_CASE(FlatPort, DivisionCameraModel)             \
  CAMERA_COMBINATION_MODEL_CASE(FlatPort, SimpleFisheyeCameraModel)        \
  CAMERA_COMBINATION_MODEL_CASE(FlatPort, FisheyeCameraModel)              \
  CAMERA_COMBINATION_MODEL_CASE(DomePort, SimplePinholeCameraModel)        \
  CAMERA_COMBINATION_MODEL_CASE(DomePort, PinholeCameraModel)              \
  CAMERA_COMBINATION_MODEL_CASE(DomePort, SimpleRadialCameraModel)         \
  CAMERA_COMBINATION_MODEL_CASE(DomePort, SimpleRadialFisheyeCameraModel)  \
  CAMERA_COMBINATION_MODEL_CASE(DomePort, RadialCameraModel)               \
  CAMERA_COMBINATION_MODEL_CASE(DomePort, RadialFisheyeCameraModel)        \
  CAMERA_COMBINATION_MODEL_CASE(DomePort, OpenCVCameraModel)               \
  CAMERA_COMBINATION_MODEL_CASE(DomePort, OpenCVFisheyeCameraModel)        \
  CAMERA_COMBINATION_MODEL_CASE(DomePort, FullOpenCVCameraModel)           \
  CAMERA_COMBINATION_MODEL_CASE(DomePort, FOVCameraModel)                  \
  CAMERA_COMBINATION_MODEL_CASE(DomePort, ThinPrismFisheyeCameraModel)     \
  CAMERA_COMBINATION_MODEL_CASE(DomePort, RadTanThinPrismFisheyeModel)     \
  CAMERA_COMBINATION_MODEL_CASE(DomePort, SimpleDivisionCameraModel)       \
  CAMERA_COMBINATION_MODEL_CASE(DomePort, DivisionCameraModel)             \
  CAMERA_COMBINATION_MODEL_CASE(DomePort, SimpleFisheyeCameraModel)        \
  CAMERA_COMBINATION_MODEL_CASE(DomePort, FisheyeCameraModel)
#endif

#define CAMERA_REFRAC_MODEL_DOES_NOT_EXIST_EXCEPTION \
  throw std::domain_error("Refractive camera model does not exist")

#define CAMERA_COMBINATION_MODEL_DOES_NOT_EXIST_EXCEPTION                     \
  throw std::domain_error(                                                    \
      "The combination of a perspective camera model and a refractive model " \
      "does not exist")

template <typename CameraRefracModel>
struct BaseCameraRefracModel {
  template <typename CameraModel, typename T>
  static inline bool IterativeProjection(
      const T* cam_params, const T* refrac_params, T u, T v, T w, T* x, T* y);
};

struct FlatPort : public BaseCameraRefracModel<FlatPort> {
  CAMERA_REFRAC_MODEL_DEFINITIONS(CameraRefracModelId::kFlatPort, "FLATPORT", 8)
};

struct DomePort : public BaseCameraRefracModel<DomePort> {
  CAMERA_REFRAC_MODEL_DEFINITIONS(CameraRefracModelId::kDomePort, "DOMEPORT", 8)
};

bool ExistsCameraRefracModelWithName(const std::string& refrac_model_name);
bool ExistsCameraRefracModelWithId(CameraRefracModelId refrac_model_id);
CameraRefracModelId CameraRefracModelNameToId(
    const std::string& refrac_model_name);
const std::string& CameraRefracModelIdToName(
    CameraRefracModelId refrac_model_id);
const std::string& CameraRefracModelParamsInfo(
    CameraRefracModelId refrac_model_id);
const std::vector<size_t>& CameraRefracModelOptimizableParamsIdxs(
    CameraRefracModelId refrac_model_id);
size_t CameraRefracModelNumParams(CameraRefracModelId refrac_model_id);
bool CameraRefracModelVerifyParams(CameraRefracModelId refrac_model_id,
                                   const std::vector<double>& params);

std::optional<Eigen::Vector2d> CameraRefracModelImgFromCam(
    CameraModelId model_id,
    CameraRefracModelId refrac_model_id,
    const std::vector<double>& cam_params,
    const std::vector<double>& refrac_params,
    const Eigen::Vector3d& uvw);

std::optional<Ray3D> CameraRefracModelCamFromImg(
    CameraModelId model_id,
    CameraRefracModelId refrac_model_id,
    const std::vector<double>& cam_params,
    const std::vector<double>& refrac_params,
    const Eigen::Vector2d& xy);

std::optional<Eigen::Vector3d> CameraRefracModelCamFromImgPoint(
    CameraModelId model_id,
    CameraRefracModelId refrac_model_id,
    const std::vector<double>& cam_params,
    const std::vector<double>& refrac_params,
    const Eigen::Vector2d& xy,
    double d);

Eigen::Vector3d CameraRefracModelRefractionAxis(
    CameraRefracModelId refrac_model_id,
    const std::vector<double>& refrac_params);

template <typename CameraRefracModel>
template <typename CameraModel, typename T>
bool BaseCameraRefracModel<CameraRefracModel>::IterativeProjection(
    const T* cam_params, const T* refrac_params, T u, T v, T w, T* x, T* y) {
  const Eigen::Matrix<T, 3, 1> uvw(u, v, w);
  const T d = uvw.norm();
  const size_t kNumIterations = 100;
  const T kMaxStepNorm = T(1e-10);
  const T kRelStepSize = T(1e-9);
  const T kAbsStepSize = T(1e-6);

  Eigen::Matrix<T, 3, 2> J;
  Eigen::Matrix<T, 2, 1> X(*x, *y);
  Eigen::Matrix<T, 3, 1> err;
  Eigen::Matrix<T, 3, 1> dx_0b;
  Eigen::Matrix<T, 3, 1> dx_0f;
  Eigen::Matrix<T, 3, 1> dx_1b;
  Eigen::Matrix<T, 3, 1> dx_1f;

  for (size_t i = 0; i < kNumIterations; ++i) {
    const T step0 = std::max(kAbsStepSize, ceres::abs(kRelStepSize * X(0)));
    const T step1 = std::max(kAbsStepSize, ceres::abs(kRelStepSize * X(1)));
    if (!CameraRefracModel::template CamFromImgPoint<CameraModel, T>(
            cam_params, refrac_params, X(0), X(1), d, &err)) {
      return false;
    }
    err -= uvw;
    if (!CameraRefracModel::template CamFromImgPoint<CameraModel, T>(
            cam_params, refrac_params, X(0) - step0, X(1), d, &dx_0b) ||
        !CameraRefracModel::template CamFromImgPoint<CameraModel, T>(
            cam_params, refrac_params, X(0) + step0, X(1), d, &dx_0f) ||
        !CameraRefracModel::template CamFromImgPoint<CameraModel, T>(
            cam_params, refrac_params, X(0), X(1) - step1, d, &dx_1b) ||
        !CameraRefracModel::template CamFromImgPoint<CameraModel, T>(
            cam_params, refrac_params, X(0), X(1) + step1, d, &dx_1f)) {
      return false;
    }

    J.col(0) = (dx_0f - dx_0b) / (T(2) * step0);
    J.col(1) = (dx_1f - dx_1b) / (T(2) * step1);
    const Eigen::Matrix<T, 2, 2> H = J.transpose() * J;
    const Eigen::Matrix<T, 2, 1> b = J.transpose() * err;
    const Eigen::Matrix<T, 2, 1> step_x = H.partialPivLu().solve(b);
    X -= step_x;
    if (step_x.squaredNorm() < kMaxStepNorm) {
      break;
    }
  }

  *x = X(0);
  *y = X(1);
  return true;
}

template <typename CameraModel, typename T>
bool FlatPort::ImgFromCam(
    const T* cam_params, const T* refrac_params, T u, T v, T w, T* x, T* y) {
  if (!CameraModel::ImgFromCam(cam_params, u, v, w, x, y)) {
    return false;
  }
  return IterativeProjection<CameraModel, T>(
      cam_params, refrac_params, u, v, w, x, y);
}

template <typename CameraModel, typename T>
bool FlatPort::CamFromImg(const T* cam_params,
                          const T* refrac_params,
                          T x,
                          T y,
                          Eigen::Matrix<T, 3, 1>* ori,
                          Eigen::Matrix<T, 3, 1>* dir) {
  *ori = Eigen::Matrix<T, 3, 1>::Zero();
  if (!CameraModel::CamFromImg(cam_params, x, y, &(*dir)(0), &(*dir)(1))) {
    return false;
  }
  (*dir)(2) = T(1);
  dir->normalize();

  const Eigen::Matrix<T, 3, 1> int_normal(
      refrac_params[0], refrac_params[1], refrac_params[2]);
  const T int_dist = refrac_params[3];
  const T int_thick = refrac_params[4];
  const T na = refrac_params[5];
  const T ng = refrac_params[6];
  const T nw = refrac_params[7];

  T d;
  if (!RayPlaneIntersection(*ori, *dir, int_normal, int_dist, &d)) {
    return false;
  }
  *ori += d * *dir;
  ComputeRefraction(int_normal, na, ng, dir);

  if (!RayPlaneIntersection(
          *ori, *dir, int_normal, T(int_dist + int_thick), &d)) {
    return false;
  }
  *ori += d * *dir;
  ComputeRefraction(int_normal, ng, nw, dir);
  return true;
}

template <typename CameraModel, typename T>
bool FlatPort::CamFromImgPoint(const T* cam_params,
                               const T* refrac_params,
                               T x,
                               T y,
                               T d,
                               Eigen::Matrix<T, 3, 1>* uvw) {
  Eigen::Matrix<T, 3, 1> ori = Eigen::Matrix<T, 3, 1>::Zero();
  Eigen::Matrix<T, 3, 1> dir = Eigen::Matrix<T, 3, 1>::Zero();
  if (!CamFromImg<CameraModel, T>(
          cam_params, refrac_params, x, y, &ori, &dir)) {
    return false;
  }

  const T lambda1 =
      -(ori.dot(dir) +
        sqrt(-ori[0] * ori[0] * dir[1] * dir[1] -
             ori[0] * ori[0] * dir[2] * dir[2] +
             T(2) * ori[0] * dir[0] * ori[1] * dir[1] +
             T(2) * ori[0] * dir[0] * ori[2] * dir[2] -
             dir[0] * dir[0] * ori[1] * ori[1] -
             dir[0] * dir[0] * ori[2] * ori[2] + dir[0] * dir[0] * d * d -
             ori[1] * ori[1] * dir[2] * dir[2] +
             T(2) * ori[1] * dir[1] * ori[2] * dir[2] -
             dir[1] * dir[1] * ori[2] * ori[2] + dir[1] * dir[1] * d * d +
             dir[2] * dir[2] * d * d)) /
      dir.dot(dir);

  const T lambda2 =
      -(ori.dot(dir) -
        sqrt(-ori[0] * ori[0] * dir[1] * dir[1] -
             ori[0] * ori[0] * dir[2] * dir[2] +
             T(2) * ori[0] * dir[0] * ori[1] * dir[1] +
             T(2) * ori[0] * dir[0] * ori[2] * dir[2] -
             dir[0] * dir[0] * ori[1] * ori[1] -
             dir[0] * dir[0] * ori[2] * ori[2] + dir[0] * dir[0] * d * d -
             ori[1] * ori[1] * dir[2] * dir[2] +
             T(2) * ori[1] * dir[1] * ori[2] * dir[2] -
             dir[1] * dir[1] * ori[2] * ori[2] + dir[1] * dir[1] * d * d +
             dir[2] * dir[2] * d * d)) /
      dir.dot(dir);

  *uvw = ori + (lambda1 >= T(0) ? lambda1 : lambda2) * dir;
  return true;
}

template <typename T>
void FlatPort::RefractionAxis(const T* refrac_params,
                              Eigen::Matrix<T, 3, 1>* refraction_axis) {
  *refraction_axis << refrac_params[0], refrac_params[1], refrac_params[2];
  refraction_axis->normalize();
}

template <typename CameraModel, typename T>
bool DomePort::ImgFromCam(
    const T* cam_params, const T* refrac_params, T u, T v, T w, T* x, T* y) {
  if (!CameraModel::ImgFromCam(cam_params, u, v, w, x, y)) {
    return false;
  }
  return IterativeProjection<CameraModel, T>(
      cam_params, refrac_params, u, v, w, x, y);
}

template <typename CameraModel, typename T>
bool DomePort::CamFromImg(const T* cam_params,
                          const T* refrac_params,
                          T x,
                          T y,
                          Eigen::Matrix<T, 3, 1>* ori,
                          Eigen::Matrix<T, 3, 1>* dir) {
  *ori = Eigen::Matrix<T, 3, 1>::Zero();
  if (!CameraModel::CamFromImg(cam_params, x, y, &(*dir)(0), &(*dir)(1))) {
    return false;
  }
  (*dir)(2) = T(1);
  dir->normalize();

  const Eigen::Matrix<T, 3, 1> sphere_center(
      refrac_params[0], refrac_params[1], refrac_params[2]);
  const T int_radius = refrac_params[3];
  const T int_thick = refrac_params[4];
  const T na = refrac_params[5];
  const T ng = refrac_params[6];
  const T nw = refrac_params[7];

  T dmin, dmax;
  int num_intersects = RaySphereIntersection(
      *ori, *dir, sphere_center, int_radius, &dmin, &dmax);
  if (num_intersects == 0) {
    return false;
  }
  *ori += dmax * *dir;

  Eigen::Matrix<T, 3, 1> normal = *ori - sphere_center;
  normal.normalize();
  ComputeRefraction(normal, na, ng, dir);

  num_intersects = RaySphereIntersection(
      *ori, *dir, sphere_center, T(int_radius + int_thick), &dmin, &dmax);
  if (num_intersects == 0) {
    return false;
  }
  *ori += dmax * *dir;
  normal = *ori - sphere_center;
  normal.normalize();
  ComputeRefraction(normal, ng, nw, dir);
  return true;
}

template <typename CameraModel, typename T>
bool DomePort::CamFromImgPoint(const T* cam_params,
                               const T* refrac_params,
                               T x,
                               T y,
                               T d,
                               Eigen::Matrix<T, 3, 1>* uvw) {
  Eigen::Matrix<T, 3, 1> ori = Eigen::Matrix<T, 3, 1>::Zero();
  Eigen::Matrix<T, 3, 1> dir = Eigen::Matrix<T, 3, 1>::Zero();
  if (!CamFromImg<CameraModel, T>(
          cam_params, refrac_params, x, y, &ori, &dir)) {
    return false;
  }

  const T lambda1 =
      -(ori.dot(dir) +
        sqrt(-ori[0] * ori[0] * dir[1] * dir[1] -
             ori[0] * ori[0] * dir[2] * dir[2] +
             T(2) * ori[0] * dir[0] * ori[1] * dir[1] +
             T(2) * ori[0] * dir[0] * ori[2] * dir[2] -
             dir[0] * dir[0] * ori[1] * ori[1] -
             dir[0] * dir[0] * ori[2] * ori[2] + dir[0] * dir[0] * d * d -
             ori[1] * ori[1] * dir[2] * dir[2] +
             T(2) * ori[1] * dir[1] * ori[2] * dir[2] -
             dir[1] * dir[1] * ori[2] * ori[2] + dir[1] * dir[1] * d * d +
             dir[2] * dir[2] * d * d)) /
      dir.dot(dir);

  const T lambda2 =
      -(ori.dot(dir) -
        sqrt(-ori[0] * ori[0] * dir[1] * dir[1] -
             ori[0] * ori[0] * dir[2] * dir[2] +
             T(2) * ori[0] * dir[0] * ori[1] * dir[1] +
             T(2) * ori[0] * dir[0] * ori[2] * dir[2] -
             dir[0] * dir[0] * ori[1] * ori[1] -
             dir[0] * dir[0] * ori[2] * ori[2] + dir[0] * dir[0] * d * d -
             ori[1] * ori[1] * dir[2] * dir[2] +
             T(2) * ori[1] * dir[1] * ori[2] * dir[2] -
             dir[1] * dir[1] * ori[2] * ori[2] + dir[1] * dir[1] * d * d +
             dir[2] * dir[2] * d * d)) /
      dir.dot(dir);

  *uvw = ori + (lambda1 >= T(0) ? lambda1 : lambda2) * dir;
  return true;
}

template <typename T>
void DomePort::RefractionAxis(const T* refrac_params,
                              Eigen::Matrix<T, 3, 1>* refraction_axis) {
  *refraction_axis << refrac_params[0], refrac_params[1], refrac_params[2];
  refraction_axis->normalize();
}

}  // namespace colmap
