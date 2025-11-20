
#include "APSS.hpp"
#include "PointCloud.hpp"
#include "Resample.hpp"
#include "Subsample.hpp"

#include <fstream>
#include <nanoflann.hpp>
#include <polyscope/implicit_helpers.h>
#include <polyscope/point_cloud.h>
#include <polyscope/polyscope.h>
#include <polyscope/types.h>
#include <polyscope/utilities.h>
#include <polyscope/volume_grid.h>
#include <stdexcept>

template <typename SDFFunc>
void addVolumeGrid(const detail::Bounds &bounds, const SDFFunc &sdf,
                   const size_t resolution = 50) {

  uint32_t dimX = resolution;
  uint32_t dimY = resolution;
  uint32_t dimZ = resolution;

  std::vector<float> values(dimX * dimY * dimZ);

  const auto xSpacing = (bounds.second.x - bounds.first.x) / (dimX - 1);
  const auto ySpacing = (bounds.second.y - bounds.first.y) / (dimY - 1);
  const auto zSpacing = (bounds.second.z - bounds.first.z) / (dimZ - 1);

  for (int ix = 0; ix < dimX; ++ix) {
    for (int iy = 0; iy < dimY; ++iy) {
      for (int iz = 0; iz < dimZ; ++iz) {
        auto p = glm::vec3(bounds.first.x + ix * xSpacing,
                           bounds.first.y + iy * ySpacing,
                           bounds.first.z + iz * zSpacing);

        const size_t idx = static_cast<size_t>(iz) * dimY * dimX +
                           static_cast<size_t>(iy) * dimX +
                           static_cast<size_t>(ix);

        values[idx] = sdf(p);
      }
    }
  }

  polyscope::VolumeGrid *psGrid = polyscope::registerVolumeGrid(
      "test grid", {dimX, dimY, dimZ}, bounds.first, bounds.second);
  psGrid->setEdgeWidth(1.0);

  polyscope::VolumeGridNodeScalarQuantity *qNode =
      psGrid->addNodeScalarQuantity("sdf node", values);
  qNode->setEnabled(true);

  qNode->setGridcubeVizEnabled(false);
  qNode->setIsosurfaceLevel(0.0);
  qNode->setIsosurfaceVizEnabled(true);
}

void drawPointCloud(const PointCloud &p) {
  auto *handCloud = polyscope::registerPointCloud("positions", p.positions);
  handCloud->addVectorQuantity("normals", p.normals);
  handCloud->setPointRadius(0.002);
  handCloud->setPointRenderMode(polyscope::PointRenderMode::Quad);
}

void writeNOFF(std::string out, const PointCloud &cloud) {
  auto ofstream = std::ofstream(out);
  ofstream << cloud.positions.size() << "\n";
  for (auto i = 0; i < cloud.positions.size(); i++) {
    auto &p = cloud.positions[i];
    auto &n = cloud.normals[i];
    ofstream << p.x << " " << p.y << " " << p.z << " " << n.x << " " << n.y
             << " " << n.z << "\n";
  }
}

PointCloud readNOFF(std::string in) {
  auto ifs = std ::ifstream(in);

  if (!ifs.is_open()) {
    throw std::runtime_error("Could not open file");
  }

  std::string line;
  std::getline(ifs, line);

  std::istringstream headerStream(line);

  size_t size;
  headerStream >> size;

  using Position = PointCloud::Position;
  using Normal = PointCloud::Normal;

  auto positions = std::vector<Position>();
  auto normals = std::vector<Normal>();

  positions.reserve(size);
  normals.reserve(size);

  for (auto i = 0; i < size; i++) {
    std::getline(ifs, line);
    std::istringstream ss(line);

    Position p;
    Normal n;
    ss >> p.x >> p.y >> p.z >> n.x >> n.y >> n.z;
    positions.push_back(p);
    normals.push_back(n);
  }

  return PointCloud(positions, normals);
}

void log_point_cloud_stats(const PointCloud &cloud) {

  const auto max_spacing = cloud.maximum_point_spacing();
  const auto avg_spacing = cloud.average_point_spacing();
  const auto min_spacing = cloud.minimum_point_spacing();

  std::cout << "max spacing: " << max_spacing << std::endl;
  std::cout << "avg spacing: " << avg_spacing << std::endl;
  std::cout << "min spacing: " << min_spacing << std::endl;
}

float cubeSDF(const glm::vec3 &p) {
  const auto b = glm::vec3(0.5f);
  glm::vec3 d = glm::abs(p) - b; // distance along each axis
  float outsideDist =
      glm::length(glm::max(d, glm::vec3(0.0f))); // distance outside cube
  float insideDist =
      std::min(std::max(d.x, std::max(d.y, d.z)), 0.0f); // negative inside
  return outsideDist + insideDist;
}

float point_spacing = 0.002f;
float sigmaP = 1.0f;
float sigmaN = 1.0f;
int gridResolution = 50;

int main() {

  polyscope::options::autocenterStructures = true;
  polyscope::view::windowWidth = 1024;
  polyscope::view::windowHeight = 1024;
  polyscope::init();

#ifdef SUBSAMPLE
  {
    const auto cloud = readNOFF("./resources/cube.txt");
    const auto cloud_subsamled =
        poissonDiskSubsample(cloud, point_spacing, sigmaP, sigmaN);
    writeNOFF("./resources/cube-sampled.txt", cloud_subsamled);
    log_point_cloud_stats(cloud);
  }
#endif

  const auto cloud = readNOFF("./resources/cube-sampled.txt");

  // // resample(handSubsampled, gaussianStd * gaussianStd, 1u);

  drawPointCloud(cloud);

  const auto apss = APSS(cloud);

  auto bounds = detail::computeBoundingBox(cloud.positions, 1.2);

  const auto apssSDF = [&apss](const glm::vec3 &p) {
    const auto fitted = apss.fit(p, point_spacing * 5);

    return std::visit([&p](const auto &fit) { return distance(fit, p); },
                      fitted);
  };

  addVolumeGrid(bounds, apssSDF, gridResolution);
  std::cout << "Done" << std::endl;

  polyscope::show();
}
