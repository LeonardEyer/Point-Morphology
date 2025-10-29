#pragma once

#include "PointCloud.hpp"
#include <Eigen/Dense>
#include <glm/glm.hpp>

struct APSS {
  const PointCloud &pointCloud;

  APSS(const PointCloud &_pointCloud) : pointCloud(_pointCloud) {};

  float evaluate_surface(const PointCloud::Position &x, float h) const {
    const float beta = 1.0f;

    static const auto phi = [](const auto &x) -> float {
      return x >= 1 ? 0 : std::pow(1 - std::pow(x, 2), 4);
    };

    static const auto weight = [h](const auto &norm) { return phi(norm / h); };

    const auto neighbours = pointCloud.knn(x, 5);

    // Weighted sums initialization
    float Sw = 0.0f;
    float Swnp = 0.0f;
    float Swpp = 0.0f;
    float SwnSwp = 0.0f;
    float SwpSwp = 0.0f;

    auto Swp = glm::vec3(0);
    auto Swn = glm::vec3(0);

    // First pass: accumulate Sw and Swp
    for (const auto &[i, norm] : neighbours) {
      auto w = weight(norm);
      Sw += w;
      Swp += w * pointCloud.positions[i];
    }

    // Second pass: accumulate remaining sums
    for (const auto &[i, norm] : neighbours) {
      auto w = weight(norm);
      auto &n = pointCloud.normals[i];
      auto &p = pointCloud.positions[i];

      Swnp += w * glm::dot(n, p);
      Swpp += w * glm::dot(p, p);
      SwnSwp += w * glm::dot(n, Swp);
      SwpSwp += w * glm::dot(p, Swp);
      Swn += w * n;
    }

    // assert(Sw != 0.0f);
     
    // Compute coefficients
    const float denom = Sw * Swpp - SwpSwp;
    float u_d1 = 0.0f;

    if (std::fabs(denom) >= 10.0f * std::numeric_limits<float>::epsilon()) {
      u_d1 = beta * 0.5f * ((Sw * Swnp - SwnSwp) / denom);
    }

    const auto u_mid = (1.0f / Sw) * (Swn - (2.0f * u_d1 * Swp));
    const float u0 = -(1.0f / Sw) * (glm::dot(u_mid, Swp) + u_d1 * Swpp);

    // Signed distance computation
    float sdval = 0.0f;

    if (std::fabs(u_d1) > 0.0f) {
      const auto c = -0.5f * (1.0f / u_d1) * u_mid;
      const float r = std::sqrt(glm::dot(c, c) - (u0 / u_d1));
      sdval = std::fabs(glm::length(c - x) - r);
    } else {
      // Degenerate case: plane
      //std::cout << "Degenerate case" << std::endl;
      sdval = std::fabs(glm::dot(x, u_mid) + u0) / glm::length(u_mid);
    }

    // Algebraic surface value (negative interior)
    float apssval = -1.0 * (u0 + glm::dot(x, u_mid) + glm::dot(x, x) * u_d1);

    // Assign correct sign
    return std::copysign(sdval, apssval);
  }
};
