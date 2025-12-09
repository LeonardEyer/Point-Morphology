#pragma once

#include <Eigen/Dense>
#include <glm/vec3.hpp>

namespace util {

inline auto concat(const glm::vec3 &a, const glm::vec3 &b) {
  return Eigen::Vector<float, 6>{a.x, a.y, a.z, b.x, b.y, b.z};
}

inline auto concat(const Eigen::Vector3f &a, const Eigen::Vector3f &b) {
  return Eigen::Vector<float, 6>{a.x(), a.y(), a.z(), b.x(), b.y(), b.z()};
}

inline glm::vec3 to_glm(const Eigen::Vector3f &x) {
  return {x.x(), x.y(), x.z()};
}

inline Eigen::Vector3f to_eigen(const glm::vec3 &x) { return {x.x, x.y, x.z}; }

using V6f = Eigen::Vector<float, 6>;
using PointNormal = V6f;
using PointNormals = std::vector<PointNormal>;

} // namespace util
