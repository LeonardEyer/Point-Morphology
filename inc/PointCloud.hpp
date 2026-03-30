#pragma once

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <happly.h>
#include <iterator>
#include <limits>
#include <memory>
#include <nanoflann.hpp>
#include <numeric>
#include <ranges>
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
      normals.emplace_back(glm::normalize(glm::vec3(nxs[i], nys[i], nzs[i])));
    }

  } catch (...) {
    throw std::runtime_error("PLY data does not contain normals");
  }
  return normals;
}

using Bounds = std::pair<glm::vec3, glm::vec3>;

inline Bounds computeBoundingBox(const std::vector<glm::vec3> &points,
                                 float margin) {
  auto min = glm::vec3(std::numeric_limits<float>::max());
  auto max = glm::vec3(-std::numeric_limits<float>::max());

  for (const auto &p : points) {
    min = glm::min(min, p);
    max = glm::max(max, p);
  }

  return Bounds(min - glm::vec3(margin), max + glm::vec3(margin));
}

inline std::vector<glm::vec3> center(const std::vector<glm::vec3> &points) {
  auto mean = std::reduce(points.cbegin(), points.cend(), glm::vec3(0.0)) /
              static_cast<float>(points.size());

  auto centerd_points = std::vector<glm::vec3>{};
  std::ranges::transform(points, std::back_inserter(centerd_points),
                         [&mean](auto &p) { return p - mean; });

  return centerd_points;
}

template <class VectorOfVectorsType, bool IsDynamic = true,
          typename num_t = float, class Distance = nanoflann::metric_L2,
          typename IndexType = size_t>
struct Vec3Adaptor {

  using self_t =
      Vec3Adaptor<VectorOfVectorsType, IsDynamic, num_t, Distance, IndexType>;

  using metric_t =
      typename Distance::template traits<num_t, self_t>::distance_t;

  using index_t =
      typename std::conditional<IsDynamic,
                                nanoflann::KDTreeSingleIndexDynamicAdaptor<
                                    metric_t, self_t, 3, IndexType>,
                                nanoflann::KDTreeSingleIndexAdaptor<
                                    metric_t, self_t, 3, IndexType>>::type;

  /** The kd-tree index for the user to call its methods as usual with any
   * other FLANN index */
  index_t *index = nullptr;

  const std::vector<glm::vec3> &pts;

  explicit Vec3Adaptor(const std::vector<glm::vec3> &points,
                       const unsigned int n_thread_build = 0)
      : pts(points) {

    index = new index_t(3, *this /* adaptor */,
                        nanoflann::KDTreeSingleIndexAdaptorParams(
                            10, nanoflann::KDTreeSingleIndexAdaptorFlags::None,
                            n_thread_build));
  }

  ~Vec3Adaptor() { delete index; }

  [[nodiscard]] inline size_t kdtree_get_point_count() const {
    return pts.size();
  }
  [[nodiscard]] inline num_t kdtree_get_pt(size_t idx, int dim) const {
    assert(idx < pts.size());
    assert(dim < 3);

    return pts[idx][dim];
  }
  template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
};

} // namespace detail

template <bool IsDynamic> struct PointKDTree {

  using Scalar = float;
  using Adaptor = detail::Vec3Adaptor<std::vector<glm::vec3>, IsDynamic>;

  Adaptor adaptor;

  PointKDTree(const std::vector<glm::vec3> &points) : adaptor(points) {};

  auto neighboursInRadius(const glm::vec3 &p, float radius,
                          bool sorted = true) const {

    Scalar query_p[3] = {p.x, p.y, p.z};
    std::vector<nanoflann::ResultItem<size_t, Scalar>> results;

    nanoflann::RadiusResultSet resultSet(radius * radius, results);
    nanoflann::SearchParameters searchParams(0, sorted);

    adaptor.index->findNeighbors(resultSet, query_p, searchParams);

    return results;
  }

  auto neighboursInRadius(const glm::vec3 &p, float radius, size_t minimum,
                          bool sorted = true) const {

    Scalar query_p[3] = {p.x, p.y, p.z};
    std::vector<nanoflann::ResultItem<size_t, Scalar>> results;

    while (results.size() <= minimum) {
      nanoflann::RadiusResultSet resultSet(radius * radius, results);
      nanoflann::SearchParameters searchParams(0, sorted);
      adaptor.index->findNeighbors(resultSet, query_p, searchParams);

      // grow radius
      radius *= 1.2;
    }

    return results;
  }

  /// Collect k nearest neighbours at p
  ///
  /// @return a vector of pairs where we have (neighbour index, squared distance
  /// to point)
  [[nodiscard]] auto knn(const glm::vec3 &p, size_t k,
                         bool sorted = true) const noexcept {

    // Collect an extra point
    k += 1;

    std::vector<size_t> ret_indexes(k);
    std::vector<Scalar> out_dists_sqr(k);

    nanoflann::KNNResultSet<float> resultSet(k);

    Scalar query_pt[3] = {p.x, p.y, p.z};
    const auto searchParams = nanoflann::SearchParameters(0, sorted);

    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
    adaptor.index->findNeighbors(resultSet, query_pt, searchParams);

    std::vector<std::pair<uint32_t, Scalar>> results(k);
    for (auto i = 0; i < k; i++) {
      results[i] = std::make_pair(ret_indexes[i], out_dists_sqr[i]);
    }

    if (results.empty()) {
      return results;
    }

    // Remove oneself from neighbours
    if (adaptor.pts[results[0].first] == p) {
      results.erase(results.begin());
    } else if (results.size() == k) {
      // remove last
      assert(sorted);
      results.erase(results.end() - 1);
    }

    return results;
  }

  void addPoints(size_t from, size_t to) const {
    if constexpr (IsDynamic) {
      adaptor.index->addPoints(from, to);
    } else {
      throw std::runtime_error("Cannot add points to static tree");
    }
  }
};

struct PointCloud {
  using Scalar = float;

  static constexpr bool IsDynamic = false;
  using KDTreeT = PointKDTree<IsDynamic>;

  using Position = glm::vec3;
  using Normal = glm::vec3;

  std::vector<Position> positions;
  std::vector<Normal> normals;
  std::unique_ptr<KDTreeT> tree;

  PointCloud(const std::vector<Position> &_positions,
             const std::vector<Normal> &_normals)
      : positions(_positions), normals(_normals) {
    tree = std::make_unique<KDTreeT>(positions);
  }

  PointCloud(std::vector<Position> &&_positions, std::vector<Normal> &&_normals)
      : positions(std::move(_positions)), normals(std::move(_normals)) {
    tree = std::make_unique<KDTreeT>(positions);
  }

  void scale(float scaling) {
    std::transform(positions.begin(), positions.end(), positions.begin(),
                   [&](auto &p) { return p * scaling; });
  }

  void translate(glm::vec3 translation) {
    std::transform(positions.begin(), positions.end(), positions.begin(),
                   [&](auto &p) { return p + translation; });
  }

  PointCloud() { tree = std::make_unique<KDTreeT>(positions); };

  // Delete copy constructor
  PointCloud(const PointCloud &other) = delete;
  PointCloud &operator=(const PointCloud &other) = delete;

  PointCloud(PointCloud &&other) noexcept
      : PointCloud(other.positions, other.normals) {
    other.tree.reset();
  }

  PointCloud &operator=(PointCloud &&other) noexcept {
    positions = std::move(other.positions);
    normals = std::move(other.normals);
    // rebuild index
    tree = std::make_unique<KDTreeT>(positions);
    other.tree.reset();
    return *this;
  }

  // Pointcloud union
  PointCloud operator&(const PointCloud &a) const noexcept {
    auto newPositions = positions;
    auto newNormals = normals;

    newPositions.insert(newPositions.end(), a.positions.begin(),
                        a.positions.end());
    newNormals.insert(newNormals.end(), a.normals.begin(), a.normals.end());
    return PointCloud(newPositions, newNormals);
  }

  [[nodiscard]] inline float getNeighbourSpacing(const Position &p,
                                                 size_t k) const {
    const auto neighbours = tree->knn(p, k);
    auto sum = 0.0f;
    for (const auto &[idx, dist_sq] : neighbours) {
      sum += std::sqrt(dist_sq);
    }
    return sum / static_cast<float>(neighbours.size());
  }

  template <typename WeightFunc>
  [[nodiscard]] inline auto
  getWeightedPoints(const Position &p, size_t k, float h,
                    const WeightFunc &weightFunc) const {

    auto weightedResult = std::vector<std::pair<size_t, float>>();

    // sorted
    const auto queryResult = tree->knn(p, k);

    // grow h until the closest neighbour has nonzero weight.
    // use sqrt so the comparison is in actual-distance space.
    while (weightFunc(h, std::sqrt(queryResult[0].second)) <
           10.f * std::numeric_limits<float>::epsilon()) {
      h *= 1.1f;
    }

    for (const auto &[i, dist_sq] : queryResult) {
      auto weight =
          weightFunc(h, std::sqrt(dist_sq)); // actual distance, not squared
      if (weight > 10.0 * std::numeric_limits<float>::epsilon()) {
        weightedResult.push_back(std::make_pair(i, weight));
      }
    }

    return weightedResult;
  }
};

inline PointCloud fromPLY(const std::string &filename) {

  auto positions = std::vector<PointCloud::Position>{};
  auto normals = std::vector<PointCloud::Normal>{};

  happly::PLYData plyIn(filename);

  const auto &vertices = plyIn.getVertexPositions();
  positions.reserve(vertices.size());
  std::transform(vertices.begin(), vertices.end(),
                 std::back_inserter(positions),
                 [](const auto &v) { return glm::vec3{v[0], v[1], v[2]}; });

  positions = detail::center(positions);

  normals = detail::getVertexNormals(plyIn);
  return {positions, normals};
}

inline auto maximum_point_spacing(const PointCloud &cloud) noexcept {
  float max_spacing = 0;

  for (const auto &p : cloud.positions) {
    const auto [_, squared_dist] = cloud.tree->knn(p, 2)[0];
    max_spacing = std::max(max_spacing, squared_dist);
  }

  return std::sqrt(max_spacing);
}

inline auto minimum_point_spacing(const PointCloud &cloud) noexcept {
  float min_spacing = std::numeric_limits<float>::max();

  for (const auto &p : cloud.positions) {
    const auto [_, squared_dist] = cloud.tree->knn(p, 2)[0];
    min_spacing = std::min(min_spacing, squared_dist);
  }

  return std::sqrt(min_spacing);
}

inline float average_point_spacing(const PointCloud &cloud) noexcept {
  float sum = 0.0f;
  size_t n = cloud.positions.size();

  for (const auto &p : cloud.positions) {
    const auto &neighbors = cloud.tree->knn(p, 1);
    float dist_sqr = neighbors[0].second;
    sum += std::sqrt(dist_sqr);
  }

  return sum / static_cast<float>(n);
}

inline PointCloud extrude(const PointCloud &p, float factor) {
  auto extrudedPoints = p.positions;
  for (auto i = 0; i < extrudedPoints.size(); i++) {
    extrudedPoints[i] += factor * p.normals[i];
  }
  return {extrudedPoints, p.normals};
}

inline Eigen::Matrix4f quadric(const PointCloud &cloud, int idx,
                               int kNeighbours = 10) {
  Eigen::Matrix4f A_qem = Eigen::Matrix4f::Zero();

  auto &x = cloud.positions[idx];
  auto neighbors = cloud.tree->knn(x, kNeighbours, false);

  for (auto j : neighbors | std::views::keys) {
    auto n = cloud.normals[j];
    auto p = cloud.positions[j];

    auto d = -glm::dot(p, n);
    auto plane = Eigen::Vector4f(n.x, n.y, n.z, d);
    A_qem += plane * plane.transpose();
  }
  return A_qem;
}

inline float curvature_estimate(const PointCloud &cloud, int idx,
                                int kNeighbours = 10) {

  auto &x = cloud.positions[idx];
  auto neighbors = cloud.tree->knn(x, kNeighbours, false);

  float max_angle = 1;
  for (auto i : neighbors | std::views::keys) {
    for (auto j : neighbors | std::views::keys) {
      auto n_i = cloud.normals[i];
      auto n_j = cloud.normals[j];
      max_angle = std::min(max_angle, glm::dot(n_i, n_j));
    }
  }

  return max_angle;
}
