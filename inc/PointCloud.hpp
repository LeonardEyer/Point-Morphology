#pragma once

#include "glm/ext/vector_float3.hpp"
#include "glm/gtc/constants.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <glm/vec3.hpp>
#include <happly.h>
#include <iterator>
#include <limits>
#include <memory>
#include <nanoflann.hpp>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace detail {

inline std::vector<glm::vec3> getVertexNormals(happly::PLYData &plyIn) {

  auto normals = std::vector<glm::vec3>{};
  try {
    std::vector<double> nxs =
        plyIn.getElement("vertex").getProperty<double>("nx");
    std::vector<double> nys =
        plyIn.getElement("vertex").getProperty<double>("ny");
    std::vector<double> nzs =
        plyIn.getElement("vertex").getProperty<double>("nz");

    normals.reserve(nxs.size());

    for (auto i = 0; i < nxs.size(); i++) {
      normals.emplace_back(nxs[i], nys[i], nzs[i]);
    }

  } catch (...) {
    throw std::runtime_error("PLY data does not contain normals");
  }
  return normals;
}

using Bounds = std::pair<glm::vec3, glm::vec3>;

inline Bounds computeBoundingBox(const std::vector<glm::vec3> &points, float scaling = 1.0f) {
  auto min = glm::vec3(std::numeric_limits<float>::infinity());
  auto max = glm::vec3(-std::numeric_limits<float>::infinity());

  for (const auto &p : points) {
    min = glm::vec3(std::min(min.x, p.x), std::min(min.y, p.y),
                    std::min(min.z, p.z));
    max = glm::vec3(std::max(max.x, p.x), std::max(max.y, p.y),
                    std::max(max.z, p.z));
  }

  return Bounds(min * scaling, max * scaling);
}

inline std::vector<glm::vec3> center(const std::vector<glm::vec3> &points) {
  auto mean = std::reduce(points.cbegin(), points.cend(), glm::vec3(0.0)) /
              static_cast<float>(points.size());

  auto centerd_points = std::vector<glm::vec3>{};
  std::ranges::transform(points, std::back_inserter(centerd_points),
                         [&mean](auto &p) { return p - mean; });

  return centerd_points;
}

struct Vec3Adaptor {
  using coord_t = float;
  const std::vector<glm::vec3> &pts;

  explicit Vec3Adaptor(const std::vector<glm::vec3> &points) : pts(points) {}

  inline size_t kdtree_get_point_count() const { return pts.size(); }
  inline coord_t kdtree_get_pt(size_t idx, int dim) const {
    if (idx >= pts.size())
      std::cout << idx << ", " << pts.size() << std::endl;
    assert(idx < pts.size());
    assert(dim < 3);

    return pts[idx][dim];
  }
  template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
};

} // namespace detail

struct PointCloud {
  using Scalar = float;

  using Position = glm::vec3;
  using Normal = glm::vec3;

  using Adaptor = detail::Vec3Adaptor;
  using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<float, Adaptor>, Adaptor, 3 /* dim */
      >;

  std::vector<Position> positions;
  std::vector<Normal> normals;

  std::unique_ptr<Adaptor> adaptor;
  std::unique_ptr<KDTree> kdTree;

  PointCloud(std::string filename) {
    happly::PLYData plyIn(filename);

    const auto &vertices = plyIn.getVertexPositions();
    positions.reserve(vertices.size());
    std::transform(vertices.begin(), vertices.end(),
                   std::back_inserter(positions),
                   [](const auto &v) { return glm::vec3{v[0], v[1], v[2]}; });

    normals = detail::getVertexNormals(plyIn);

    positions = detail::center(positions);
    
    // KDTree construction
    adaptor = std::make_unique<Adaptor>(positions);
    kdTree = std::make_unique<KDTree>(3, *adaptor);
  }

  inline auto neighboursInRadius(const Position &p,
                                 float radius) const noexcept {
    std::vector<nanoflann::ResultItem<uint32_t, Scalar>> results;

    Scalar query_p[3] = {p.x, p.y, p.z};

    kdTree->radiusSearch(&query_p[0], radius, results);

    return results;
  }

  inline auto knn(const Position &p, size_t k) const noexcept {

    std::vector<uint32_t> ret_indexes(k);
    std::vector<Scalar> out_dists_sqr(k);

    nanoflann::KNNResultSet<float> resultSet(k);

    Scalar query_pt[3] = {p.x, p.y, p.z};

    auto num_results =
        kdTree->knnSearch(&query_pt[0], k, &ret_indexes[0], &out_dists_sqr[0]);
    assert(num_results == k);

    std::vector<std::pair<uint32_t, Scalar>> results(k);
    for (auto i = 0; i < k; i++) {
      results[i] = std::make_pair(ret_indexes[i], out_dists_sqr[i]);
    }
    return results;
  }

  template <typename WeightFunc>
  inline auto getWeightedPoints(const Position &p, size_t k,
                                const WeightFunc &weightFunc) const {

    auto weightedResult = std::vector<std::pair<size_t, float>>();

    for (const auto &[i, norm] : knn(p, k)) {
      auto weight = weightFunc(norm);
      if (weight > 10.0 * std::numeric_limits<float>::epsilon()) {
	weightedResult.push_back(std::make_pair(i, weight));
      }
    }

    return weightedResult;
  }
  
};
