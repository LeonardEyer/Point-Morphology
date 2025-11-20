#pragma once

#include "APSS.hpp"
#include "PointCloud.hpp"
#include <cassert>
#include <functional>
#include <glm/glm.hpp>

namespace structuring_elements {

namespace sdf {

const auto sphere = [](const glm::vec3 &p) { return glm::length(p) - 1.0f; };

const auto torus = [](const glm::vec3 &p) {
  static const auto t = glm::vec2(1.0, 0.3);
  glm::vec2 q = glm::vec2(glm::length(glm::vec2(p.x, p.z)) - t.x, p.y);
  return glm::length(q) - t.y;
};

const auto cube = [](const glm::vec3 &p) {
  const auto b = glm::vec3(0.5f);
  glm::vec3 d = glm::abs(p) - b; // distance along each axis
  float outsideDist =
      glm::length(glm::max(d, glm::vec3(0.0f))); // distance outside cube
  float insideDist =
      std::min(std::max(d.x, std::max(d.y, d.z)), 0.0f); // negative inside
  return outsideDist + insideDist;
};

} // namespace sdf

struct PointStructuringElement {

  using Position = glm::vec3;
  using SDF = std::function<float(const Position &)>;

  float s;    // scale
  Position c; // center
  SDF B;      // sdf of its shape

  PointStructuringElement(float _s, Position _c, SDF _B)
      : s(_s), c(_c), B(_B) {};

  float distance(const Position &x) const noexcept {
    return s * B((x - c) / s);
  }

  Position gradient(const Position &x) const noexcept {
    // approximation of gradient (for now)
    const float eps = 1e-4f; // adjust if necessary

    // basis
    const Position ex(eps, 0, 0);
    const Position ey(0, eps, 0);
    const Position ez(0, 0, eps);

    float dx = (distance(x + ex) - distance(x - ex)) / (2.0f * eps);
    float dy = (distance(x + ey) - distance(x - ey)) / (2.0f * eps);
    float dz = (distance(x + ez) - distance(x - ez)) / (2.0f * eps);

    return glm::normalize(Position(dx, dy, dz));
  }
};

inline PointStructuringElement fit(const APSS &pss,
                                   const PointCloud::Position &x,
                                   const PointStructuringElement::SDF &sdf,
                                   float scale) {

  // We assume that this point is far enough away from our surface

  // We aim to minimize (9)

  // 1. collect a point sampling PI of the implicit surface
  const auto neighbours = pss.pointCloud.knn(x, 2);

  assert(neighbours.size() > 2);

  // 2. initialize mean shift with n (usually 2) meaningful points
  // {c_j^0} := {closest points in PI under PSE distance}
  const auto candidate = [&] {
    std::vector<std::pair<float, size_t>> distances;
    distances.reserve(neighbours.size());
    for (const auto &[idx, _] : neighbours) {
      auto xi = pss.pointCloud.positions[idx];
      auto pse = PointStructuringElement{scale, xi, sdf};
      distances.emplace_back(pse.distance(x), idx);
    }
    std::ranges::nth_element(
        distances, distances.begin() + 1,
        [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    return PointStructuringElement{
        scale, pss.pointCloud.positions[distances.front().second], sdf};
  }();

  return candidate;
  // TODO mean shift
  // TODO choose global minimizer from our converged points
}

inline PointCloud::Position project(const PointStructuringElement &pse,
                                    const PointCloud::Position &p) {
  return p - pse.distance(p) * pse.gradient(p);
}

inline PointCloud::Position
project_iterative(const APSS &pss, const PointCloud::Position &x,
                  const PointStructuringElement::SDF &sdf, float scale) {

  static constexpr auto threshold = 1e-8f;
  static constexpr auto maxIter = 100;

  const auto P = [&pss, &sdf, scale](const auto &p) {
    return project(fit(pss, p, sdf, scale), p);
  };

  auto xi = x;
  auto xip1 = P(xi);

  for (auto i = 0; i < maxIter; i++) {

    if (glm::distance(xi, xip1) < threshold * threshold) {
      // converged
      break;
    }

    xi = xip1;
    xip1 = P(xip1);
  }

  return xip1;
}

} // namespace structuring_elements
