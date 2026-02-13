#pragma once

#include "PointCloud.hpp"
#include "Utils.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include <limits>
#include <stdexcept>
#include <variant>

struct FitParams {
  float u0;
  glm::vec3 u_mid;
  float u_d1;

  FitParams(float _u0, const glm::vec3 &_u_mid, float _u_d1)
      : u0(_u0), u_mid(_u_mid), u_d1(_u_d1) {}

  [[nodiscard]] constexpr int
  apssign(const PointCloud::Position &x) const noexcept {
    return std::copysign(1, u0 + glm::dot(x, u_mid) + glm::dot(x, x) * u_d1);
  }
};

struct FitResult {
  FitParams params;
  inline explicit FitResult(const FitParams &_params) : params(_params) {}
};

struct SphereFitResult : FitResult {
  glm::vec3 center{};
  float radius;

  explicit SphereFitResult(const FitParams &_params) : FitResult(_params) {
    center = -0.5f * (1.0f / params.u_d1) * params.u_mid;
    radius = std::sqrt(glm::dot(center, center) - (params.u0 / params.u_d1));
  }
};

struct PlaneFitResult : FitResult {
  glm::vec3 normal{}; // normalized plane normal
  float d;            // plane offset (n·x + d = 0)

  explicit PlaneFitResult(const FitParams &_params)
      : FitResult(_params), normal(glm::normalize(_params.u_mid)),
        d(_params.u0) {}
};

inline float distance(const SphereFitResult &fit,
                                const PointCloud::Position &x) noexcept {
  auto sdval = std::fabs(glm::length(fit.center - x) - fit.radius);
  return sdval * fit.params.apssign(x);
}

inline float distance(const PlaneFitResult &fit,
                                const PointCloud::Position &x) noexcept {
  auto sdval = std::fabs(glm::dot(x, fit.params.u_mid) + fit.params.u0) /
               glm::length(fit.params.u_mid);
  return sdval * fit.params.apssign(x);
}

inline PointCloud::Position gradient(const SphereFitResult &fit,
                                     const PointCloud::Position &x) {
  auto v = x - fit.center;

  if (glm::length2(v) == 0.0f) {
    throw std::runtime_error("Degenerate gradient for SphereFitResult");
  }
  v = glm::normalize(v);

  // if we go in the gradient direction we expect a positive sign?.
  // i guess this only makes sense if we expect x to be close to the surface
  if (fit.params.apssign(x + v) < 0) {
    v *= -1;
  }

  return v;
}

inline PointCloud::Position gradient(const PlaneFitResult &fit,
                                     const PointCloud::Position &p) {
  auto n = fit.normal;
  float len = glm::length(n);

  if (len > 0.0f) {
    return n / len;
  }
  throw std::runtime_error("Degenerate gradient for PlaneFitResult");
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

  [[nodiscard]] FitVariant fit(const PointCloud::Position &x, float h) const {
    constexpr double beta = 1.0;

    const auto spacing = pointCloud.getNeighbourSpacing(x, 6);

    static const auto phi = [](const auto &x) -> float {
      return x >= 1 ? 0 : std::pow(1 - x * x, 4);
    };

    // reference to h
    const auto weight = [spacing](float h, const auto &norm) {
      return phi(norm / (h * spacing));
    };

    const auto neighbours = pointCloud.getWeightedPoints(x, 20, h, weight);

    float minDistSq = neighbours[0].second;
    float maxAllowedDist = h * spacing * 2.0f; 

    if (minDistSq > (maxAllowedDist * maxAllowedDist)) {
      
      auto n = pointCloud.normals[neighbours[0].first];
      auto p = pointCloud.positions[neighbours[0].first];

      FitParams params{-glm::dot(n, p), n, 0};

      return PlaneFitResult(params);
    }

    // Weighted sums initialization
    double Sw = 0.0f;
    double Swnp = 0.0f;
    double Swpp = 0.0f;
    double SwnSwp = 0.0f;
    double SwpSwp = 0.0f;

    Eigen::Vector3d Swp = Eigen::Vector3d::Zero();
    Eigen::Vector3d Swn = Eigen::Vector3d::Zero();

    // First pass: accumulate Sw and Swp
    for (const auto &[i, w] : neighbours) {
      Sw += w;
      Swp += w * util::to_eigen(pointCloud.positions[i]).cast<double>();
    }

    // Second pass: accumulate remaining sums
    for (const auto &[i, w] : neighbours) {

      auto n = util::to_eigen(pointCloud.normals[i]).cast<double>();
      auto p = util::to_eigen(pointCloud.positions[i]).cast<double>();

      Swnp += w * n.dot(p);
      Swpp += w * p.dot(p);
      SwnSwp += w * n.dot(Swp);
      SwpSwp += w * p.dot(Swp);
      Swn += w * n;
    }

    assert(Sw != 0.0f);

    // Compute coefficients
    const auto denom = Sw * Swpp - SwpSwp;

    double u_d1 = std::abs(denom) >= std::numeric_limits<double>::epsilon()
                      ? beta * 0.5f * ((Sw * Swnp - SwnSwp) / denom)
                      : 0.0;

    const auto u_mid = (1.0f / Sw) * (Swn - (2.0f * u_d1 * Swp));
    const float u0 = -(1.0f / Sw) * (u_mid.dot(Swp) + u_d1 * Swpp);

    FitParams params{u0, util::to_glm(u_mid.cast<float>()),
                     static_cast<float>(u_d1)};

    if (std::fabs(u_d1) > 1e-4f) {
      return SphereFitResult(params);
    } else {
      return PlaneFitResult(params);
    }
  }

  inline float evaluate_surface(const PointCloud::Position &x, float h) const {
    return std::visit([&](const auto &fit) { return distance(fit, x); },
                      fit(x, h));
  }

  inline PointCloud::Normal evaluate_gradient(const PointCloud::Position &x,
                                              float h) const {
    return std::visit([&](const auto &fit) { return gradient(fit, x); },
                      fit(x, h));
  }
};

inline std::pair<PointCloud::Position, PointCloud::Normal>
project_iterative(const APSS &pss, const PointCloud::Position &x, float scale,
                  size_t maxIter = 100) {
  const auto spacing = pss.pointCloud.getNeighbourSpacing(x, 6);
  const auto eps = 1e-4f * spacing;

  const auto P = [&pss, h = scale, eps](const auto &p) {
    auto fitted = pss.fit(p, h);
    return std::visit(
        [&p](const auto &fitvariant) { return project(fitvariant, p); },
        fitted);
  };

  auto xi = x;
  auto xip1 = P(xi);

  for (auto i = 0; i < maxIter; i++) {

    if (!std::isfinite(glm::distance2(xi, xip1)))
      break;

    if (glm::distance2(xi, xip1) <= (eps * eps))
      break;

    xi = xip1;
    xip1 = P(xip1);
  }

  const auto normal = pss.evaluate_gradient(x, scale);

  return {xip1, normal};
}
