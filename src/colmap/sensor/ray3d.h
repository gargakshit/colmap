#pragma once

#include "colmap/util/eigen_alignment.h"

#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

struct Ray3D {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Ray3D();
  Ray3D(const Eigen::Vector3d& ori, const Eigen::Vector3d& dir);

  Eigen::Vector3d At(double distance) const;

  Eigen::Vector3d ori;
  Eigen::Vector3d dir;
};

template <typename T>
inline void ComputeRefraction(const Eigen::Matrix<T, 3, 1>& normal,
                              T n1,
                              T n2,
                              Eigen::Matrix<T, 3, 1>* v);

template <typename T>
inline int RaySphereIntersection(const Eigen::Matrix<T, 3, 1>& ray_ori,
                                 const Eigen::Matrix<T, 3, 1>& ray_dir,
                                 const Eigen::Matrix<T, 3, 1>& center,
                                 T r,
                                 T* dmin,
                                 T* dmax);

template <typename T>
inline bool RayPlaneIntersection(const Eigen::Matrix<T, 3, 1>& ray_ori,
                                 const Eigen::Matrix<T, 3, 1>& ray_dir,
                                 const Eigen::Matrix<T, 3, 1>& normal,
                                 T dist,
                                 T* d);

template <typename T>
inline T PointToRayDistance(const Eigen::Matrix<T, 3, 1>& point,
                            const Eigen::Matrix<T, 3, 1>& ray_ori,
                            const Eigen::Matrix<T, 3, 1>& ray_dir);

template <typename T>
inline bool IntersectLinesWithTolerance(const Eigen::Matrix<T, 3, 1>& origin1,
                                        const Eigen::Matrix<T, 3, 1>& dir1,
                                        const Eigen::Matrix<T, 3, 1>& origin2,
                                        const Eigen::Matrix<T, 3, 1>& dir2,
                                        Eigen::Matrix<T, 3, 1>& intersection,
                                        T tolerance = T(1e-8));

template <typename T>
inline void ComputeRefraction(const Eigen::Matrix<T, 3, 1>& normal,
                              T n1,
                              T n2,
                              Eigen::Matrix<T, 3, 1>* v) {
  if (n1 == n2) {
    return;
  }

  const T r = n1 / n2;
  const T c = normal.dot(*v);
  const T scale = (r * c - sqrt(T(1.0) - r * r * (T(1.0) - c * c)));
  *v = r * *v - scale * normal;
  (*v).normalize();
}

template <typename T>
inline int RaySphereIntersection(const Eigen::Matrix<T, 3, 1>& ray_ori,
                                 const Eigen::Matrix<T, 3, 1>& ray_dir,
                                 const Eigen::Matrix<T, 3, 1>& center,
                                 T r,
                                 T* dmin,
                                 T* dmax) {
  const Eigen::Matrix<T, 3, 1> diff = center - ray_ori;
  const T t0 = diff.dot(ray_dir);
  const T d_squared = diff.dot(diff) - t0 * t0;
  if (d_squared > r * r) {
    return 0;
  }
  const T t1 = sqrt(r * r - d_squared);
  *dmin = t0 - t1;
  *dmax = t0 + t1;
  return 2;
}

template <typename T>
inline bool RayPlaneIntersection(const Eigen::Matrix<T, 3, 1>& ray_ori,
                                 const Eigen::Matrix<T, 3, 1>& ray_dir,
                                 const Eigen::Matrix<T, 3, 1>& normal,
                                 T dist,
                                 T* d) {
  const Eigen::Matrix<T, 3, 1> p0 = dist * normal;
  const T denom = ray_dir.dot(normal);
  if (ceres::abs(denom) < std::numeric_limits<T>::epsilon()) {
    return false;
  }

  *d = (p0 - ray_ori).dot(normal) / denom;
  return true;
}

template <typename T>
inline T PointToRayDistance(const Eigen::Matrix<T, 3, 1>& point,
                            const Eigen::Matrix<T, 3, 1>& ray_ori,
                            const Eigen::Matrix<T, 3, 1>& ray_dir) {
  const T t = (point - ray_ori).dot(ray_dir);
  const Eigen::Matrix<T, 3, 1> point_closest = ray_ori + t * ray_dir;
  return (point_closest - point).norm();
}

template <typename T>
inline bool IntersectLinesWithTolerance(const Eigen::Matrix<T, 3, 1>& origin1,
                                        const Eigen::Matrix<T, 3, 1>& dir1,
                                        const Eigen::Matrix<T, 3, 1>& origin2,
                                        const Eigen::Matrix<T, 3, 1>& dir2,
                                        Eigen::Matrix<T, 3, 1>& intersection,
                                        T tolerance) {
  const Eigen::Matrix<T, 3, 1> o = origin1 - origin2;
  const T a = dir1.dot(dir1);
  const T b = dir1.dot(dir2);
  const T c = dir2.dot(dir2);
  const T d = dir1.dot(o);
  const T e = dir2.dot(o);
  const T denom = b * b - a * c;
  if (ceres::abs(denom) <= std::numeric_limits<T>::epsilon()) {
    intersection = origin1;
    return true;
  }

  const T s = (c * d - b * e) / denom;
  const T t = (b * d - a * e) / denom;
  const Eigen::Matrix<T, 3, 1> pa = origin1 + s * dir1;
  const Eigen::Matrix<T, 3, 1> pb = origin2 + t * dir2;
  if ((pb - pa).norm() >= tolerance) {
    return false;
  }

  intersection = pa + (pb - pa) / T(2);
  return true;
}

}  // namespace colmap
