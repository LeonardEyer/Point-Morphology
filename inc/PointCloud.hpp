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

template <class VectorOfVectorsType, typename num_t = float,
          class Distance = nanoflann::metric_L2, typename IndexType = size_t>
struct Vec3Adaptor {

  using self_t = Vec3Adaptor<VectorOfVectorsType, num_t, Distance, IndexType>;
  using metric_t =
      typename Distance::template traits<num_t, self_t>::distance_t;
  using index_t = nanoflann::KDTreeSingleIndexDynamicAdaptor<metric_t, self_t,
                                                             3, IndexType>;

  /** The kd-tree index for the user to call its methods as usual with any
   * other FLANN index */
  index_t *index = nullptr;

  const std::vector<glm::vec3> &pts;

  explicit Vec3Adaptor(const std::vector<glm::vec3> &points,
                       const unsigned int n_thread_build = 1)
      : pts(points) {

    index = new index_t(
        3, *this /* adaptor */,
        nanoflann::KDTreeSingleIndexAdaptorParams(
            2, nanoflann::KDTreeSingleIndexAdaptorFlags::None, n_thread_build));
  }

  ~Vec3Adaptor() { delete index; }

  [[nodiscard]] inline size_t kdtree_get_point_count() const {
    return pts.size();
  }
  [[nodiscard]] inline num_t kdtree_get_pt(size_t idx, int dim) const {
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

struct PointKDTree {

  using Scalar = float;
  using Adaptor = detail::Vec3Adaptor<std::vector<glm::vec3>>;

  Adaptor adaptor;

  PointKDTree(const std::vector<glm::vec3> &points) : adaptor(points, 8) {};

  auto neighboursInRadius(const glm::vec3 &p, float radius,
                          bool sorted = true) const {

    Scalar query_p[3] = {p.x, p.y, p.z};
    std::vector<nanoflann::ResultItem<size_t, Scalar>> results;

    nanoflann::RadiusResultSet resultSet(radius * radius, results);
    nanoflann::SearchParameters searchParams(0, sorted);

    adaptor.index->findNeighbors(resultSet, query_p, searchParams);

    // Remove self from results
    if (!results.empty() && adaptor.pts[results[0].first] == p) {
      results.erase(results.begin());
    }

    return results;
  }

  auto knn(const glm::vec3 &p, size_t k, bool sorted = true) const noexcept {

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
    adaptor.index->addPoints(from, to);
  }
};

struct PointCloud {
  using Scalar = float;

  using Position = glm::vec3;
  using Normal = glm::vec3;

  std::vector<Position> positions;
  std::vector<Normal> normals;
  std::unique_ptr<PointKDTree> tree;

  PointCloud(const std::vector<Position> &_positions,
             const std::vector<Normal> &_normals)
      : positions(_positions), normals(_normals) {
    tree = std::make_unique<PointKDTree>(positions);
  }

  PointCloud(std::vector<Position> &&_positions, std::vector<Normal> &&_normals)
      : positions(std::move(_positions)), normals(std::move(_normals)) {
    tree = std::make_unique<PointKDTree>(positions);
  }

  void scale(float scaling) {
    std::transform(positions.begin(), positions.end(), positions.begin(),
                   [scaling](auto &p) { return p * scaling; });
  }

  PointCloud() { tree = std::make_unique<PointKDTree>(positions); };

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
    tree = std::make_unique<PointKDTree>(positions);
    other.tree.reset();
    return *this;
  }

  void insertPoint(const Position &p, const Normal &n) {
    positions.push_back(p);
    normals.push_back(n);
    const auto newIndex = positions.size() - 1;
    tree->addPoints(newIndex, newIndex);
  }

  void updatePoint(size_t index, const Position &p, const Normal &n) {
    positions[index] = p;
    normals[index] = n;

    tree->adaptor.index->removePoint(index);
    tree->addPoints(index, index);
    // tree = std::make_unique<PointKDTree>(positions);
  }

  void deletePoint(size_t index) { assert(false); }

  template <typename WeightFunc>
  inline auto getWeightedPoints(const Position &p, size_t k,
                                const WeightFunc &weightFunc) const {

    auto weightedResult = std::vector<std::pair<size_t, float>>();

    for (const auto &[i, norm] : tree->knn(p, k)) {
      auto weight = weightFunc(norm);
      if (weight > 10.0 * std::numeric_limits<float>::epsilon()) {
        weightedResult.push_back(std::make_pair(i, weight));
      }
    }

    return weightedResult;
  }
};

inline PointCloud fromPLY(std::string filename) {

  auto positions = std::vector<PointCloud::Position>{};
  auto normals = std::vector<PointCloud::Normal>{};

  happly::PLYData plyIn(filename);

  const auto &vertices = plyIn.getVertexPositions();
  positions.reserve(vertices.size());
  std::transform(vertices.begin(), vertices.end(),
                 std::back_inserter(positions),
                 [](const auto &v) { return glm::vec3{v[0], v[1], v[2]}; });

  normals = detail::getVertexNormals(plyIn);
  return PointCloud(positions, normals);
}

inline auto maximum_point_spacing(const PointCloud &cloud) noexcept {
  float max_spacing = 0;

  for (const auto &p : cloud.positions) {
    const auto [_, dist] = cloud.tree->knn(p, 2)[0];
    max_spacing = std::max(max_spacing, dist);
  }

  return std::sqrt(max_spacing);
}

inline auto minimum_point_spacing(const PointCloud &cloud) noexcept {
  float min_spacing = std::numeric_limits<float>::max();

  for (const auto &p : cloud.positions) {
    const auto [_, dist] = cloud.tree->knn(p, 2)[0];
    min_spacing = std::min(min_spacing, dist);
  }

  return std::sqrt(min_spacing);
}

inline auto average_point_spacing(const PointCloud &cloud) noexcept {
  float avg_spacing = 0;
  for (const auto &p : cloud.positions) {
    const auto [_, dist] = cloud.tree->knn(p, 2)[0];
    avg_spacing += dist;
  }
  avg_spacing /= (cloud.positions.size() * 2);
  return std::sqrt(avg_spacing);
}

inline PointCloud extrude(const PointCloud &p, float factor) {
  auto extrudedPoints = p.positions;
  for (auto i = 0; i < extrudedPoints.size(); i++) {
    extrudedPoints[i] += factor * p.normals[i];
  }
  return PointCloud(extrudedPoints, p.normals);
}

inline Eigen::Matrix4f quadric(const PointCloud &cloud, int idx,
                               int kNeighbours = 10) {
  Eigen::Matrix4f A_qem = Eigen::Matrix4f::Zero();

  auto &x = cloud.positions[idx];
  auto neighbors = cloud.tree->knn(x, kNeighbours, false);

  for (auto [j, _] : neighbors) {
    auto n = cloud.normals[j];
    auto p = cloud.positions[j];

    auto d = -glm::dot(p, n);
    auto plane = Eigen::Vector4f(n.x, n.y, n.z, d);
    // std::cout << "plane = \n" << plane << std::endl;
    A_qem += plane * plane.transpose();
  }
  return A_qem;
}

inline float curvature_estimate(const PointCloud &cloud, int idx,
                                int kNeighbours = 10) {

  auto &x = cloud.positions[idx];
  auto neighbors = cloud.tree->knn(x, kNeighbours, false);

  float max_angle = 1;
  for (auto [i, _] : neighbors) {
    for (auto [j, _] : neighbors) {
      auto n_i = cloud.normals[i];
      auto n_j = cloud.normals[j];
      max_angle = std::min(max_angle, glm::dot(n_i, n_j));
    }
  }

  return max_angle;
}
