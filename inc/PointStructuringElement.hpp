#pragma once

#include <functional>
#include <glm/glm.hpp>

namespace structuring_elements {

auto sphereSDF = [](const glm::vec3 &p) { return glm::length(p) - 1.0f; };

auto torusSDF = [](const glm::vec3 &p) {
  static const auto t = glm::vec2(1.0, 0.3);
  glm::vec2 q = glm::vec2(glm::length(glm::vec2(p.x, p.z)) - t.x, p.y);
  return glm::length(q) - t.y;
};

struct PointStructuringElement {

  using Position = glm::vec3;
  using SDF = std::function<float(const Position &)>;

  float s;    // scale
  Position c; // center
  SDF B;      // sdf of its shape

  PointStructuringElement(float _s, Position _c, SDF _B)
      : s(_s), c(_c), B(_B) {};

  float operator()(const Position &x) const noexcept {
    return s * B((x - c) / s);
  }
};

} // namespace structuring_elements
