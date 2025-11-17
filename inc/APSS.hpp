#pragma once

#include "PointCloud.hpp"
#include "glm/ext/vector_float3.hpp"
#include <Eigen/Dense>
#include <glm/glm.hpp>
#include <limits>
#include <variant>

template <typename T> constexpr int sgn(T val) noexcept {
  return (T(0) < val) - (val < T(0));
}

struct FitParams {
  float u0;
  glm::vec3 u_mid;
  float u_d1;

  FitParams(float _u0, const glm::vec3 &_u_mid, float _u_d1)
      : u0(_u0), u_mid(_u_mid), u_d1(_u_d1) {}

  [[nodiscard]] constexpr int
  apssign(const PointCloud::Position &x) const noexcept {
    return sgn(u0 + glm::dot(x, u_mid) + glm::dot(x, x) * u_d1);
  }
};

struct FitResult {
  FitParams params;
  inline FitResult(const FitParams &_params) : params(_params) {}
};

struct SphereFitResult : FitResult {
  glm::vec3 center;
  float radius;

  SphereFitResult(const FitParams &_params) : FitResult(_params) {
    center = -0.5f * (1.0f / params.u_d1) * params.u_mid;
    radius = std::sqrt(glm::dot(center, center) - (params.u0 / params.u_d1));
  }
};

struct PlaneFitResult : FitResult {
  PlaneFitResult(const FitParams &_params) : FitResult(_params) {}
};

inline constexpr float distance(const SphereFitResult &fit,
                                const PointCloud::Position &x) noexcept {
  auto sdval = std::fabs(glm::length(fit.center - x) - fit.radius);
  return sdval * fit.params.apssign(x);
}

inline constexpr float distance(const PlaneFitResult &fit,
                                const PointCloud::Position &x) noexcept {
  auto sdval = std::fabs(glm::dot(x, fit.params.u_mid) + fit.params.u0) /
               glm::length(fit.params.u_mid);
  return sdval * fit.params.apssign(x);
}

inline PointCloud::Position gradient(const SphereFitResult &fit,
                                     const PointCloud::Position &x) noexcept {
  glm::vec3 v = x - fit.center;
  float len = glm::length(v);
  if (len > 0.0f) {
    return v / len; // unit vector pointing away from center
  } else {
    return glm::vec3(1.0f, 0.0f, 0.0f); // arbitrary direction
  }
}

inline PointCloud::Position gradient(const PlaneFitResult &fit,
                                     const PointCloud::Position &p) noexcept {
  glm::vec3 n = fit.params.u_mid;
  float len = glm::length(n);
  return (len > 0.0f) ? n / len
                      : glm::vec3(1.0f, 0.0f, 0.0f); // arbitrary if degenerate
}

template <typename T>
inline constexpr PointCloud::Position
project(const T &fit, const PointCloud::Position &p) noexcept {
  float dist = distance(fit, p);     // signed distance
  glm::vec3 grad = gradient(fit, p); // unit gradient
  return p - dist * grad;            // projected point
}

struct APSS {

  const PointCloud &pointCloud;

  APSS(const PointCloud &_pointCloud) : pointCloud(_pointCloud) {};

  using FitVariant = std::variant<SphereFitResult, PlaneFitResult>;

  FitVariant fit(const PointCloud::Position &x, float h) const {
    const float beta = 1.0f;

    static const auto phi = [](const auto &x) -> float {
      return x >= 1 ? 0 : std::pow(1 - x * x, 4);
    };

    // reference to h
    const auto weight = [&h](const auto &norm) { return phi(norm / h); };

    auto neighbours = pointCloud.getWeightedPoints(x, 5, weight);

    while (neighbours.empty()) {
      h *= 2;
      neighbours = pointCloud.getWeightedPoints(x, 5, weight);
    }

    // Weighted sums initialization
    float Sw = 0.0f;
    float Swnp = 0.0f;
    float Swpp = 0.0f;
    float SwnSwp = 0.0f;
    float SwpSwp = 0.0f;

    auto Swp = glm::vec3(0);
    auto Swn = glm::vec3(0);

    // First pass: accumulate Sw and Swp
    for (const auto &[i, w] : neighbours) {
      Sw += w;
      Swp += w * pointCloud.positions[i];
    }

    // Second pass: accumulate remaining sums
    for (const auto &[i, w] : neighbours) {

      auto &n = pointCloud.normals[i];
      auto &p = pointCloud.positions[i];

      Swnp += w * glm::dot(n, p);
      Swpp += w * glm::dot(p, p);
      SwnSwp += w * glm::dot(n, Swp);
      SwpSwp += w * glm::dot(p, Swp);
      Swn += w * n;
    }

    assert(Sw != 0.0f);

    // Compute coefficients
    const float denom = Sw * Swpp - SwpSwp;
    float u_d1 = 0.0f;

    if (std::fabs(denom) >= 10.0f * std::numeric_limits<float>::epsilon()) {
      u_d1 = beta * 0.5f * ((Sw * Swnp - SwnSwp) / denom);
    }

    const auto u_mid = (1.0f / Sw) * (Swn - (2.0f * u_d1 * Swp));
    const float u0 = -(1.0f / Sw) * (glm::dot(u_mid, Swp) + u_d1 * Swpp);

    FitParams params{u0, u_mid, u_d1};

    if (std::fabs(u_d1) > 0.0f) {
      return SphereFitResult(params);
    } else {
      return PlaneFitResult(params);
    }
  }

  inline float evaluate_surface(const PointCloud::Position &x, float h) const {
    return std::visit([&](const auto &fit) { return distance(fit, x); },
                      fit(x, h));
  }
};
