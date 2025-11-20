#pragma once

#include <algorithm>
#include <cstddef>
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

inline Bounds computeBoundingBox(const std::vector<glm::vec3> &points,
                                 float scaling = 1.0f) {
  auto min = glm::vec3(std::numeric_limits<float>::max());
  auto max = glm::vec3(-std::numeric_limits<float>::max());

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
    if (idx >= pts.size()) {
      throw std::runtime_error("Out of bounds kdtree access");
    }
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
  using KDTree = nanoflann::KDTreeSingleIndexDynamicAdaptor<
      nanoflann::L2_Simple_Adaptor<float, Adaptor>, Adaptor, 3 /* dim */
      >;

  std::vector<Position> positions;
  std::vector<Normal> normals;

  // adaptor will hold a const ref to our data points
  std::unique_ptr<Adaptor> adaptor;
  // kdtree is used for fast indexing on the adaptor data
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

  PointCloud(const std::vector<Position> &_positions,
             const std::vector<Normal> &_normals) {
    positions = detail::center(_positions);
    normals = _normals;

    // KDTree construction
    adaptor = std::make_unique<Adaptor>(positions);
    kdTree = std::make_unique<KDTree>(3, *adaptor);
  }

  PointCloud() {
    // KDTree construction
    adaptor = std::make_unique<Adaptor>(positions);
    kdTree = std::make_unique<KDTree>(3, *adaptor);
  }

  inline auto insertPoint(const Position &p, const Normal &n) {
    positions.push_back(p);
    normals.push_back(n);
    const auto newIndex = positions.size() - 1;
    kdTree->addPoints(newIndex, newIndex);
  }

  inline auto deletePoint(size_t index) { kdTree->removePoint(index); }

  inline auto neighboursInRadius(const Position &p, float radius,
                                 bool sorted = true) const noexcept {

    Scalar query_p[3] = {p.x, p.y, p.z};

    const auto searchParams = nanoflann::SearchParameters(0, sorted);

    const Scalar radiusSqr = radius * radius;

    std::vector<nanoflann::ResultItem<size_t, Scalar>> results;
    nanoflann::RadiusResultSet<Scalar, size_t> resultSet(radiusSqr, results);
    kdTree->findNeighbors(resultSet, query_p, searchParams);

    // The point is not its own neighbour
    if (not results.empty() && positions[results[0].first] == p) {
      // std::cout << "myself" << std::endl;
      // std::cout << "neighbours = " << results.size() << std::endl;
      results.erase(results.begin());
    }

    return results;
  }

  inline auto knn(const Position &p, size_t k,
                  bool sorted = true) const noexcept {

    std::vector<size_t> ret_indexes(k);
    std::vector<Scalar> out_dists_sqr(k);

    nanoflann::KNNResultSet<float> resultSet(k);

    Scalar query_pt[3] = {p.x, p.y, p.z};
    const auto searchParams = nanoflann::SearchParameters(0, sorted);

    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

    kdTree->findNeighbors(resultSet, query_pt, searchParams);

    std::vector<std::pair<uint32_t, Scalar>> results(k);
    for (auto i = 0; i < k; i++) {
      results[i] = std::make_pair(ret_indexes[i], out_dists_sqr[i]);
    }

    // Remove oneself from neighbours
    if (not results.empty() && positions[results[0].first] == p) {
      results.erase(results.begin());
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

  inline auto maximum_point_spacing() const noexcept {
    float max_spacing = 0;

    for (const auto &p : positions) {
      const auto [_, dist] = knn(p, 2)[0];
      max_spacing = std::max(max_spacing, dist);
    }

    return std::sqrt(max_spacing);
  }

  inline auto minimum_point_spacing() const noexcept {
    float min_spacing = std::numeric_limits<float>::max();

    for (const auto &p : positions) {
      const auto [_, dist] = knn(p, 2)[0];
      min_spacing = std::min(min_spacing, dist);
    }

    return std::sqrt(min_spacing);
  }

  inline auto average_point_spacing() const noexcept {
    float avg_spacing = 0;
    for (const auto &p : positions) {
      const auto [_, dist] = knn(p, 2)[0];
      avg_spacing += dist;
    }
    avg_spacing /= (positions.size() * 2);
    return std::sqrt(avg_spacing);
  }

  inline auto poisson_disk_subsample() noexcept {}
};
