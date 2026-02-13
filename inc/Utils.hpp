#pragma once

#include <Eigen/Dense>
#include <glm/vec3.hpp>

namespace util {

  template<typename Func>
glm::vec3 gradient(const glm::vec3 &x, const Func &func, float eps = 1) {
   
    // basis
    const auto ex = glm::vec3(eps, 0, 0);
    const auto ey = glm::vec3(0, eps, 0);
    const auto ez = glm::vec3(0, 0, eps);

    float dx = (func(x + ex) - func(x - ex)) / (2.0f * eps);
    float dy = (func(x + ey) - func(x - ey)) / (2.0f * eps);
    float dz = (func(x + ez) - func(x - ez)) / (2.0f * eps);

    return glm::vec3(dx, dy, dz);
}

struct SSGAdaptor {
  using VectorType = glm::vec3;
  using DataType = std::vector<VectorType>;
  using IndexType = size_t;

  static float get_x(const VectorType &v) { return v[0]; }
  static float get_y(const VectorType &v) { return v[1]; }
  static float get_z(const VectorType &v) { return v[2]; }
  static const VectorType &get_vec(const DataType &data, IndexType idx) {
    return data[idx];
  }
  static float distance2(const VectorType &a, const VectorType &b) {
    float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
  }

  static IndexType insert(DataType &d, const VectorType &v) {
    d.push_back(v);
    return d.size() - 1;
  }

  static void update(DataType &d, IndexType idx, const VectorType &v) {
    d[idx] = v;
  }
};

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
using Feature6D = V6f;
using Features6D = std::vector<Feature6D>;

} // namespace util
