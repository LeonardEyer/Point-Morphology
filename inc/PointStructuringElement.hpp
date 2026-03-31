#pragma once

#include "APSS.hpp"
#include "PointCloud.hpp"
#include "Utils.hpp"
#include "glm/common.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <glm/ext/vector_float3.hpp>
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <optional>
#include <polyscope/utilities.h>

namespace structuring_elements {

namespace sdf {

const auto sphere = [](const glm::vec3 &p) { return glm::length(p) - 1.0f; };

const auto torus = [](const glm::vec3 &p) {
  static const auto t =
      glm::vec2(0.3846, 0.1154); // scaled for 1x1x1 bounding volume
  glm::vec2 q = glm::vec2(glm::length(glm::vec2(p.x, p.z)) - t.x, p.y);
  return glm::length(q) - t.y;
};

const auto cube = [](const glm::vec3 &p) {
  const auto b = glm::vec3(1.f); // / sqrt(3));
  glm::vec3 d = glm::abs(p) - b; // distance along each axis
  float outsideDist =
      glm::length(glm::max(d, glm::vec3(0.0f))); // distance outside cube
  float insideDist =
      std::min(std::max(d.x, std::max(d.y, d.z)), 0.0f); // negative inside
  return outsideDist + insideDist;
};

const auto roundcube = [](const glm::vec3 &p) {
  static const auto b = glm::vec3(1.f); // sqrt(3));
  static const auto r = .1f;
  const auto q = glm::abs(p) - b + r;
  return glm::length(glm::max(q, glm::vec3(0.0f))) +
         std::min(std::max(q.x, std::max(q.y, q.z)), 0.0f) - r;
};

} // namespace sdf

struct PointStructuringElement {

  using Position = glm::vec3;
  using Jacobian = glm::mat3;
  using SDF = std::function<float(const Position &)>;

  float s;    // scale
  Position c; // center
  SDF B;      // sdf of its shape

  std::optional<Jacobian> c_grad; // gradient of c wrt to the fitted position x?

  PointStructuringElement(float _s, Position _c, SDF _B,
                          std::optional<Jacobian> _c_grad = std::nullopt)
      : s(_s), c(_c), B(_B), c_grad(_c_grad) {};

  [[nodiscard]] float distance(const Position &x) const noexcept {
    return s * B((x - c) / s);
  }

  [[nodiscard]] Position gradient(const Position &x) const noexcept {
    // approximation of gradient (for now)
    constexpr float eps = 1e-4f; // adjust if necessary

    // basis
    constexpr Position ex(eps, 0, 0);
    constexpr Position ey(0, eps, 0);
    constexpr Position ez(0, 0, eps);

    float dx = (distance(x + ex) - distance(x - ex)) / (2.0f * eps);
    float dy = (distance(x + ey) - distance(x - ey)) / (2.0f * eps);
    float dz = (distance(x + ez) - distance(x - ez)) / (2.0f * eps);

    return glm::normalize(Position(dx, dy, dz));
  }
};

inline auto mean_shift(const glm::vec3 &x,
                       const std::vector<glm::vec3> &candidates,
                       const std::vector<glm::vec3> &neighbours,
                       const PointStructuringElement::SDF &sdf,
                       float pse_scale) {
  static constexpr auto sqEps = 1e-4 * 1e-4;
  static constexpr auto maxIter = 20u;
  static constexpr auto initial_grad = glm::mat3(1.0f);

  // ensure that the largest possible point in search radius is close to zero
  // weighting (3 stddev of gaussian)
  const auto scale = pse_scale / 3.f;

  const auto weightFunc = [scaleSq = scale * scale](const auto &x) -> float {
    return std::exp(-x / (2.0f * scaleSq));
  };

  auto new_candiates = candidates;
  auto new_canidates_grads = std::vector(candidates.size(), initial_grad);

  for (auto j = 0; j < new_candiates.size(); j++) {
    auto &c_j = new_candiates[j];
    auto &c_j_grad = new_canidates_grads[j];

    auto diff = std::numeric_limits<float>::max();
    auto iter = 0u;

    while (diff > sqEps && iter < maxIter) {
      auto c_j_k = glm::vec3(0.);
      auto c_j_k_grad = glm::mat3(0.0f);
      auto denominator = 0.0f;
      auto sum_weight_grad_theta = glm::vec3(0.);

      for (const auto &p_i : neighbours) {

        const auto B = PointStructuringElement{pse_scale, p_i, sdf};
        const auto B_p = B.distance(x) + pse_scale;
        const auto B_p_grad = B.gradient(x);

        // w_i^{k-1} (A9)
        const auto weight =
            weightFunc(glm::length2(c_j - p_i)) * weightFunc(std::pow(B_p, 2));

        // \nabla \theta_i^{k-1}
        glm::vec3 grad_theta = (-2.0f / (scale * scale)) *
                               ((c_j - p_i) * c_j_grad + B_p * B_p_grad);
        c_j_k += weight * p_i;
        c_j_k_grad += weight * glm::outerProduct(p_i, grad_theta);
        sum_weight_grad_theta += weight * grad_theta;
        denominator += weight;
      }

      c_j_k_grad -= glm::outerProduct(c_j, sum_weight_grad_theta);

      // mean
      // divide by zero check
      if (denominator >= 1e-9f) {
        c_j_k /= denominator;
        c_j_k_grad /= denominator;
      }

      diff = glm::distance2(c_j_k, c_j);
      c_j = c_j_k;
      c_j_grad = c_j_k_grad;

      // Frobenius Normalization
      float norm = glm::length(glm::vec3(glm::length(c_j_grad[0]),
                                         glm::length(c_j_grad[1]),
                                         glm::length(c_j_grad[2])));
      if (norm > 1e-9f) {
        c_j_grad /= norm;
      }

      iter++;
    }
  }
  return std::make_pair(new_candiates, new_canidates_grads);
}

inline PointStructuringElement fit(const APSS &pss,
                                   const PointCloud::Position &x,
                                   const PointStructuringElement::SDF &sdf,
                                   float scale, bool verbose = false) {

  // We aim to minimize (9)
  auto searchRadius = scale;
  // 1. collect a point sampling PI of the implicit surface
  const auto neighbours = [&]() {
    std::vector<PointCloud::Position> neighbours;
    auto result = pss.pointCloud.tree->neighboursInRadius(x, scale);

    while (result.size() < 10) {
      searchRadius *= 1.1;
      result = pss.pointCloud.tree->neighboursInRadius(x, searchRadius);
    }

    for (const auto &[idx, _] : result) {
      neighbours.emplace_back(pss.pointCloud.positions[idx]);
    }
    return neighbours;
  }();

  // 2. initialize mean shift with n (usually 2) meaningful points
  // {c_j^0} := {closest points in PI under PSE distance}
  const auto cj0 = [&] {
    std::vector<std::pair<float, glm::vec3>> distances;
    distances.reserve(neighbours.size());
    for (const auto &xi : neighbours) {
      auto pse = PointStructuringElement{scale, xi, sdf};
      distances.emplace_back(pse.distance(x), xi);
    }

    // we are looking at the 2 most meaningful points
    auto nth = 2;
    std::ranges::nth_element(
        distances, distances.begin() + nth - 1,
        [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    auto cjs = std::vector<glm::vec3>(nth);
    for (auto i = 0; i < nth; i++) {
      cjs[i] = distances[i].second;
    }
    return cjs;
  }();

  const auto [converged_candidates, converged_grads] =
      mean_shift(x, cj0, neighbours, sdf, searchRadius);

  // Choose Global Minimizer (Eq 11)
  auto best_energy = std::numeric_limits<float>::max();
  PointCloud::Position global_c = converged_candidates[0];
  PointStructuringElement::Jacobian global_c_grad = converged_grads[0];

  for (auto i = 0u; i < converged_candidates.size(); i++) {
    auto &cand_c = converged_candidates[i];
    auto &cand_c_grad = converged_grads[i];
    // Evaluate the PSE distance (B_c(x)) for this candidate
    auto pse = PointStructuringElement{scale, cand_c, sdf};
    float energy = std::abs(pse.distance(x));

    if (energy < best_energy) {
      best_energy = energy;
      global_c = cand_c;
      global_c_grad = cand_c_grad;
    }
  }

  return PointStructuringElement{scale, global_c, sdf, global_c_grad};
}

inline PointCloud::Position project(const PointStructuringElement &pse,
                                    const PointCloud::Position &p) {
  // How do we handle the adjoint case if pse is not spherical??
  return p - pse.distance(p) * pse.gradient(p);
}

enum Operation { Dilation, Erosion };

template <Operation op>
PointCloud::Position shift(const PointStructuringElement &,
                           const PointCloud::Position &, bool);

template <>
inline PointCloud::Position
shift<Operation::Dilation>(const PointStructuringElement &pse,
                           const PointCloud::Position &p, bool inside) {

  static constexpr auto eps = 1e-4f;

  // Only shift if we are inside
  if (!inside) {
    return p;
  }

  const auto distance = pse.c - p;
  const auto e_m =
      (1.f + pse.s) + eps; // Move to center (1) + to bounding sphere (s)

  const auto delta = e_m * glm::normalize(distance);

  return p + delta;
}

template <>
inline PointCloud::Position
shift<Operation::Erosion>(const PointStructuringElement &pse,
                          const PointCloud::Position &p, bool inside) {

  static constexpr auto eps = 1e-4f;

  // Only shift if we are outside
  if (inside) {
    return p;
  }

  const auto distance = pse.c - p;
  const auto e_m =
      (1.f + pse.s) + eps; // Move to center (1) + to bounding sphere (s)

  const auto delta = e_m * glm::normalize(distance);

  return p + delta;
}

template <Operation op>
std::pair<PointCloud::Position, PointCloud::Normal>
project_iterative(const APSS &pss, const PointCloud::Position &x,
                  const PointStructuringElement::SDF &sdf, float pse_scale,
                  unsigned maxIter = 100, float threshold = 1e-4f) {

  const auto P = [&](const auto &p) {
    auto fitted = fit(pss, p, sdf, pse_scale);

    // compute indicator function for shift procedure
    // TODO: Speed up indicator function computation
    // remark: we add a bufer zone of the pse_scale to make sure that we are not
    // "inside" the PSE
    auto inside = pss.evaluate_surface(p) <= (pse_scale / 2);

    // shift then project
    return project(fitted, shift<op>(fitted, p, inside));
  };

  auto xi = x;
  auto xip1 = P(xi);

  for (auto i = 0; i < maxIter; i++) {

    if (glm::distance2(xi, xip1) <= threshold * threshold) {
      // converged
      break;
    }

    xi = xip1;
    xip1 = P(xip1);
  }

  const auto fitted = fit(pss, xip1, sdf, pse_scale);
  const auto J_c = fitted.c_grad;

  auto normal = glm::vec3();

  if (J_c.has_value()) {

    auto grad_B = fitted.gradient(xip1);

    // Erosion means flipping the pse normal
    if constexpr (op == Erosion) {
      grad_B *= -1;
    }
    static constexpr glm::mat3 Identity(1.0f);

    // This term ensures that even if grad_B is [1,0,0],
    // it is transformed by the relationship between the point and the fit.
    normal = glm::normalize(grad_B * (Identity - *J_c));

  } else {
    std::cout << "woops" << std::endl;
    // const auto variational_morphology = [&](const auto &p) {
    //   const auto pse_distance =
    //       structuring_elements::fit(pss, p, sdf, pse_scale).distance(p);
    //   const auto apss_distance = pss.evaluate_surface(p, pss_scale);
    //   if (op == structuring_elements::Erosion) {
    //     return std::max(-pse_distance, apss_distance);
    //   }
    //   return std::min(pse_distance, apss_distance);
    // };
    // auto normal = glm::normalize(util::gradient(x, variational_morphology,
    // 1));
    normal = glm::normalize(util::gradient(
        x, [&](const auto &x) { return fitted.distance(x); }, 0.75));

    if constexpr (op == Erosion) {
      normal *= -1;
    }
  }
  return {xip1, normal};
}

} // namespace structuring_elements
