#include "colmap/sensor/models_refrac.h"

#include <unordered_map>

namespace colmap {

#define CAMERA_REFRAC_MODEL_CASE(CameraRefracModel)                      \
  constexpr CameraRefracModelId CameraRefracModel::refrac_model_id;      \
  const std::string CameraRefracModel::refrac_model_name =               \
      CameraRefracModel::InitializeRefracModelName();                    \
  constexpr size_t CameraRefracModel::num_params;                        \
  const std::string CameraRefracModel::refrac_params_info =              \
      CameraRefracModel::InitializeRefracParamsInfo();                   \
  const std::vector<size_t> CameraRefracModel::optimizable_params_idxs = \
      CameraRefracModel::InitializeOptimizableParamsIdxs();

CAMERA_REFRAC_MODEL_CASES

#undef CAMERA_REFRAC_MODEL_CASE

namespace {

std::unordered_map<std::string, CameraRefracModelId>
InitializeCameraRefracModelNameToId() {
  std::unordered_map<std::string, CameraRefracModelId> result;
#define CAMERA_REFRAC_MODEL_CASE(CameraRefracModel) \
  result.emplace(CameraRefracModel::refrac_model_name, CameraRefracModel::refrac_model_id);
  CAMERA_REFRAC_MODEL_CASES
#undef CAMERA_REFRAC_MODEL_CASE
  return result;
}

std::unordered_map<CameraRefracModelId, const std::string*>
InitializeCameraRefracModelIdToName() {
  std::unordered_map<CameraRefracModelId, const std::string*> result;
#define CAMERA_REFRAC_MODEL_CASE(CameraRefracModel) \
  result.emplace(CameraRefracModel::refrac_model_id, &CameraRefracModel::refrac_model_name);
  CAMERA_REFRAC_MODEL_CASES
#undef CAMERA_REFRAC_MODEL_CASE
  return result;
}

const std::unordered_map<std::string, CameraRefracModelId>
    kCameraRefracModelNameToId = InitializeCameraRefracModelNameToId();
const std::unordered_map<CameraRefracModelId, const std::string*>
    kCameraRefracModelIdToName = InitializeCameraRefracModelIdToName();

}  // namespace

bool ExistsCameraRefracModelWithName(const std::string& refrac_model_name) {
  return kCameraRefracModelNameToId.count(refrac_model_name) > 0;
}

bool ExistsCameraRefracModelWithId(CameraRefracModelId refrac_model_id) {
  return kCameraRefracModelIdToName.count(refrac_model_id) > 0;
}

CameraRefracModelId CameraRefracModelNameToId(
    const std::string& refrac_model_name) {
  const auto it = kCameraRefracModelNameToId.find(refrac_model_name);
  return it == kCameraRefracModelNameToId.end() ? CameraRefracModelId::kInvalid
                                                : it->second;
}

const std::string& CameraRefracModelIdToName(
    CameraRefracModelId refrac_model_id) {
  const auto it = kCameraRefracModelIdToName.find(refrac_model_id);
  if (it == kCameraRefracModelIdToName.end()) {
    static const std::string kEmptyRefracModelName;
    return kEmptyRefracModelName;
  }
  return *it->second;
}

const std::string& CameraRefracModelParamsInfo(
    CameraRefracModelId refrac_model_id) {
  switch (refrac_model_id) {
    case CameraRefracModelId::kFlatPort:
      return FlatPort::refrac_params_info;
    case CameraRefracModelId::kDomePort:
      return DomePort::refrac_params_info;
    case CameraRefracModelId::kInvalid:
      break;
  }
  static const std::string kEmptyParamsInfo;
  return kEmptyParamsInfo;
}

const std::vector<size_t>& CameraRefracModelOptimizableParamsIdxs(
    CameraRefracModelId refrac_model_id) {
  switch (refrac_model_id) {
    case CameraRefracModelId::kFlatPort:
      return FlatPort::optimizable_params_idxs;
    case CameraRefracModelId::kDomePort:
      return DomePort::optimizable_params_idxs;
    case CameraRefracModelId::kInvalid:
      break;
  }
  static const std::vector<size_t> kEmptyIdxs;
  return kEmptyIdxs;
}

size_t CameraRefracModelNumParams(CameraRefracModelId refrac_model_id) {
  switch (refrac_model_id) {
    case CameraRefracModelId::kFlatPort:
      return FlatPort::num_params;
    case CameraRefracModelId::kDomePort:
      return DomePort::num_params;
    case CameraRefracModelId::kInvalid:
      break;
  }
  return 0;
}

bool CameraRefracModelVerifyParams(CameraRefracModelId refrac_model_id,
                                   const std::vector<double>& params) {
  if (refrac_model_id == CameraRefracModelId::kInvalid) {
    return params.empty();
  }
  return params.size() == CameraRefracModelNumParams(refrac_model_id);
}

std::string FlatPort::InitializeRefracParamsInfo() {
  return "Nx, Ny, Nz, int_dist, int_thick, na, ng, nw";
}

std::vector<size_t> FlatPort::InitializeOptimizableParamsIdxs() {
  return {0, 1, 2, 3};
}

std::string DomePort::InitializeRefracParamsInfo() {
  return "Cx, Cy, Cz, int_radius, int_thick, na, ng, nw";
}

std::vector<size_t> DomePort::InitializeOptimizableParamsIdxs() {
  return {0, 1, 2};
}

std::optional<Eigen::Vector2d> CameraRefracModelImgFromCam(
    CameraModelId model_id,
    CameraRefracModelId refrac_model_id,
    const std::vector<double>& cam_params,
    const std::vector<double>& refrac_params,
    const Eigen::Vector3d& uvw) {
  Eigen::Vector2d xy;
#define CAMERA_COMBINATION_MODEL_CASE(CameraRefracModel, CameraModel)           \
  if (model_id == CameraModel::model_id &&                                      \
      refrac_model_id == CameraRefracModel::refrac_model_id) {                  \
    if (CameraRefracModel::ImgFromCam<CameraModel>(cam_params.data(),           \
                                                   refrac_params.data(),        \
                                                   uvw.x(),                     \
                                                   uvw.y(),                     \
                                                   uvw.z(),                     \
                                                   &xy.x(),                     \
                                                   &xy.y())) {                  \
      return xy;                                                                \
    }                                                                           \
    return std::nullopt;                                                        \
  } else

  CAMERA_COMBINATION_MODEL_CASES {
    CAMERA_COMBINATION_MODEL_DOES_NOT_EXIST_EXCEPTION;
  }

#undef CAMERA_COMBINATION_MODEL_CASE
  return std::nullopt;
}

std::optional<Ray3D> CameraRefracModelCamFromImg(
    CameraModelId model_id,
    CameraRefracModelId refrac_model_id,
    const std::vector<double>& cam_params,
    const std::vector<double>& refrac_params,
    const Eigen::Vector2d& xy) {
  Ray3D ray;
#define CAMERA_COMBINATION_MODEL_CASE(CameraRefracModel, CameraModel)           \
  if (model_id == CameraModel::model_id &&                                      \
      refrac_model_id == CameraRefracModel::refrac_model_id) {                  \
    if (CameraRefracModel::CamFromImg<CameraModel>(cam_params.data(),           \
                                                   refrac_params.data(),        \
                                                   xy.x(),                      \
                                                   xy.y(),                      \
                                                   &ray.ori,                    \
                                                   &ray.dir)) {                 \
      return ray;                                                               \
    }                                                                           \
    return std::nullopt;                                                        \
  } else

  CAMERA_COMBINATION_MODEL_CASES {
    CAMERA_COMBINATION_MODEL_DOES_NOT_EXIST_EXCEPTION;
  }

#undef CAMERA_COMBINATION_MODEL_CASE
  return std::nullopt;
}

std::optional<Eigen::Vector3d> CameraRefracModelCamFromImgPoint(
    CameraModelId model_id,
    CameraRefracModelId refrac_model_id,
    const std::vector<double>& cam_params,
    const std::vector<double>& refrac_params,
    const Eigen::Vector2d& xy,
    double d) {
  Eigen::Vector3d uvw;
#define CAMERA_COMBINATION_MODEL_CASE(CameraRefracModel, CameraModel)           \
  if (model_id == CameraModel::model_id &&                                      \
      refrac_model_id == CameraRefracModel::refrac_model_id) {                  \
    if (CameraRefracModel::CamFromImgPoint<CameraModel>(cam_params.data(),      \
                                                        refrac_params.data(),   \
                                                        xy.x(),                 \
                                                        xy.y(),                 \
                                                        d,                      \
                                                        &uvw)) {                \
      return uvw;                                                               \
    }                                                                           \
    return std::nullopt;                                                        \
  } else

  CAMERA_COMBINATION_MODEL_CASES {
    CAMERA_COMBINATION_MODEL_DOES_NOT_EXIST_EXCEPTION;
  }

#undef CAMERA_COMBINATION_MODEL_CASE
  return std::nullopt;
}

Eigen::Vector3d CameraRefracModelRefractionAxis(
    CameraRefracModelId refrac_model_id,
    const std::vector<double>& refrac_params) {
  Eigen::Vector3d refrac_axis = Eigen::Vector3d::UnitZ();
  switch (refrac_model_id) {
    case CameraRefracModelId::kFlatPort:
      FlatPort::RefractionAxis(refrac_params.data(), &refrac_axis);
      break;
    case CameraRefracModelId::kDomePort:
      DomePort::RefractionAxis(refrac_params.data(), &refrac_axis);
      break;
    case CameraRefracModelId::kInvalid:
      break;
  }
  return refrac_axis;
}

}  // namespace colmap
