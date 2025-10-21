#include <algorithm>
#include <glm/geometric.hpp>
#include <happly.h>
#include <polyscope/implicit_helpers.h>
#include <polyscope/point_cloud.h>
#include <polyscope/polyscope.h>
#include <polyscope/types.h>
#include <polyscope/utilities.h>
#include <polyscope/volume_grid.h>

struct PointSetSurface {
  using Position = glm::vec3;
  using Normal = glm::vec3;

  std::vector<Position> positions;
  std::vector<Normal> normals;

  PointSetSurface(std::string filename) {
    happly::PLYData plyIn(filename);

    const auto &vertices = plyIn.getVertexPositions();
    positions.reserve(vertices.size());
    std::transform(vertices.begin(), vertices.end(),
                   std::back_inserter(positions),
                   [](const auto &v) { return glm::vec3{v[0], v[1], v[2]}; });

    // --- Load normals if present ---
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
      std::cout << "No normals found in PLY file.\n";
    }

    std::cout << "Loaded " << positions.size() << " positions";
    if (!normals.empty())
      std::cout << " with normals";
    std::cout << ".\n";
  }
};

auto torusSDF = [](glm::vec3 p) {
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

void callback() {
  if (ImGui::Button("add implicits")) {
    polyscope::ImplicitRenderOpts opts;
    polyscope::ImplicitRenderMode mode =
        polyscope::ImplicitRenderMode::SphereMarch;
    opts.subsampleFactor = 2; // downsample the rendering

    // render the implicit isosurface from the current viewport
    polyscope::DepthRenderImageQuantity *img =
        polyscope::renderImplicitSurface("torus sdf", torusSDF, mode, opts);
  }
}

int main() {

  // Options
  polyscope::options::autocenterStructures = true;
  polyscope::view::windowWidth = 1024;
  polyscope::view::windowHeight = 1024;

  // Initialize polyscope
  polyscope::init();

  const auto hand = PointSetSurface("./resources/hand.ply");

  auto *handCloud =
      polyscope::registerPointCloud("hand positions", hand.positions);

  handCloud->addVectorQuantity("normals", hand.normals);

  handCloud->setPointRadius(0.0002);
  handCloud->setPointRenderMode(polyscope::PointRenderMode::Quad);

  // addVolumeGrid();

  polyscope::state::userCallback = callback;

  polyscope::show();
}
