#include <algorithm>
#include <glm/geometric.hpp>
#include <happly.h>
#include <nanoflann.hpp>
#include <polyscope/implicit_helpers.h>
#include <polyscope/point_cloud.h>
#include <polyscope/polyscope.h>
#include <polyscope/types.h>
#include <polyscope/utilities.h>
#include <polyscope/volume_grid.h>
#include <stdexcept>

std::vector<glm::vec3> getVertexNormals(happly::PLYData &plyIn) {

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

struct PointCloud {
  using coord_t = float;

  using Position = glm::vec3;
  using Normal = glm::vec3;

  std::vector<Position> positions;
  std::vector<Normal> normals;

  PointCloud(std::string filename) {
    happly::PLYData plyIn(filename);

    const auto &vertices = plyIn.getVertexPositions();
    positions.reserve(vertices.size());
    std::transform(vertices.begin(), vertices.end(),
                   std::back_inserter(positions),
                   [](const auto &v) { return glm::vec3{v[0], v[1], v[2]}; });

    normals = getVertexNormals(plyIn);
  }
};

// And this is the "dataset to kd-tree" adaptor class:
template <typename Derived> struct PointCloudAdaptor {
  using coord_t = typename Derived::coord_t;

  const Derived &obj; //!< A const ref to the data set origin

  /// The constructor that sets the data set source
  PointCloudAdaptor(const Derived &obj_) : obj(obj_) {}

  /// CRTP helper method
  inline const Derived &derived() const { return obj; }

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const {
    return derived().positions.size();
  }

  // Returns the dim'th component of the idx'th point in the class:
  inline coord_t kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (dim == 0)
      return derived().positions[idx].x;
    else if (dim == 1)
      return derived().positions[idx].y;
    else
      return derived().positions[idx].z;
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop.
  template <class BBOX> bool kdtree_get_bbox(BBOX & /*bb*/) const {
    return false;
  }
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

auto sphereSDF = [](const glm::vec3 &p) { return glm::length(p) - 1.0f; };

auto torusSDF = [](const glm::vec3 &p) {
  // TODO: Make a generator for different values of t
  static const auto t = glm::vec2(1.0, 0.3);
  glm::vec2 q = glm::vec2(glm::length(glm::vec2(p.x, p.z)) - t.x, p.y);
  return glm::length(q) - t.y;
};

void addVolumeGrid() {

  uint32_t dimX = 50;
  uint32_t dimY = 50;
  uint32_t dimZ = 50;

  glm::vec3 bound_low{-3., -3., -3.};
  glm::vec3 bound_high{3., 3., 3.};

  polyscope::VolumeGrid *psGrid = polyscope::registerVolumeGrid(
      "test grid", {dimX, dimY, dimZ}, bound_low, bound_high);

  psGrid->setEdgeWidth(1.0);

  polyscope::VolumeGridNodeScalarQuantity *qNode =
      psGrid->addNodeScalarQuantityFromCallable("torus sdf node", torusSDF);
  qNode->setEnabled(true);

  qNode->setGridcubeVizEnabled(false);
  qNode->setIsosurfaceLevel(0.0);
  qNode->setIsosurfaceVizEnabled(true);
}

int main() {

  // Options
  polyscope::options::autocenterStructures = true;
  polyscope::view::windowWidth = 1024;
  polyscope::view::windowHeight = 1024;

  // Initialize polyscope
  polyscope::init();

  const auto hand = PointCloud("./resources/hand.ply");

  using PC2KD = PointCloudAdaptor<PointCloud>;
  const PC2KD pc2kd(hand); // The adaptor
                           // construct a kd-tree index:
  using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<float, PC2KD>, PC2KD, 3 /* dim */
      >;

  std::cout << "Build kdtree" << std::endl;
  my_kd_tree_t index(3, pc2kd);

  const auto neighoursInRadius = [&index](const auto &p, const auto radius) {
    const auto search_radius = static_cast<float>(radius);
    std::vector<nanoflann::ResultItem<uint32_t, float>> ret_matches;
    float query_pt[3] = {p.x, p.y, p.z};

    const size_t nMatches =
        index.radiusSearch(&query_pt[0], search_radius, ret_matches);

    return ret_matches;
  };
  std::cout << "Collecting neighbours" << std::endl;
  auto result = neighoursInRadius(glm::vec3(0.5, 0.5, 0.5), 3.5f);

  std::cout << "Collected " << result.size()  << " neighbours" << std::endl;
  // for (const auto &r : result) {
  //   std::cout << "result(" << r.first << ", " << r.second << ")" << std::endl;
  // }

  auto *handCloud =
      polyscope::registerPointCloud("hand positions", hand.positions);

  handCloud->addVectorQuantity("normals", hand.normals);

  handCloud->setPointRadius(0.0002);
  handCloud->setPointRenderMode(polyscope::PointRenderMode::Quad);

  // addVolumeGrid();

  polyscope::show();
}
