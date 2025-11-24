
#include "APSS.hpp"
#include "PointCloud.hpp"
#include "PointStructuringElement.hpp"
#include "Resample.hpp"
#include "Subsample.hpp"
#include "polyscope/pick.h"

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

void drawPointCloud(std::string name, const PointCloud &p) {
  auto *handCloud = polyscope::registerPointCloud(name, p.positions);
  handCloud->addVectorQuantity("normals", p.normals);
  handCloud->setPointRadius(0.002);
  handCloud->setPointRenderMode(polyscope::PointRenderMode::Quad);
}

void drawPointCloud(std::string name,
                    const std::vector<PointCloud::Position> &p) {
  auto *handCloud = polyscope::registerPointCloud(name, p);
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

std::vector<glm::vec3> sampleBoxSurface(const detail::Bounds &bounds,
                                        int numSamples) {
  std::vector<glm::vec3> samples;

  glm::vec3 min = bounds.first;
  glm::vec3 max = bounds.second;

  // Compute box dimensions
  glm::vec3 size = max - min;

  // Approximate samples per edge
  int samplesPerEdge =
      std::ceil(std::cbrt(numSamples / 3.0f)); // rough cube root

  // Loop over faces (6 faces)
  for (int face = 0; face < 6; ++face) {
    for (int i = 0; i < samplesPerEdge; ++i) {
      for (int j = 0; j < samplesPerEdge; ++j) {
        float u = i / float(samplesPerEdge - 1);
        float v = j / float(samplesPerEdge - 1);

        glm::vec3 point;

        switch (face) {
        case 0:
          point = glm::vec3(min.x, min.y + u * size.y, min.z + v * size.z);
          break; // left
        case 1:
          point = glm::vec3(max.x, min.y + u * size.y, min.z + v * size.z);
          break; // right
        case 2:
          point = glm::vec3(min.x + u * size.x, min.y, min.z + v * size.z);
          break; // bottom
        case 3:
          point = glm::vec3(min.x + u * size.x, max.y, min.z + v * size.z);
          break; // top
        case 4:
          point = glm::vec3(min.x + u * size.x, min.y + v * size.y, min.z);
          break; // back
        case 5:
          point = glm::vec3(min.x + u * size.x, min.y + v * size.y, max.z);
          break; // front
        }

        samples.push_back(point);

        if (samples.size() >= numSamples) {
          return samples; // stop early if we reached target
        }
      }
    }
  }

  return samples;
}

std::vector<glm::vec3> sampleBoxVolume(const detail::Bounds &bounds,
                                       int numSamples) {
  std::vector<glm::vec3> samples;

  glm::vec3 min = bounds.first;
  glm::vec3 max = bounds.second;

  glm::vec3 size = max - min;

  // Compute the number of points along each axis
  int pointsPerAxis =
      std::ceil(std::cbrt(float(numSamples))); // cube root for volume
  if (pointsPerAxis < 1)
    pointsPerAxis = 1;

  glm::vec3 step = size / float(pointsPerAxis - 1); // spacing between points

  for (int x = 0; x < pointsPerAxis; ++x) {
    for (int y = 0; y < pointsPerAxis; ++y) {
      for (int z = 0; z < pointsPerAxis; ++z) {
        glm::vec3 point = min + glm::vec3(x, y, z) * step;
        samples.push_back(point);

        if (samples.size() >= numSamples) {
          return samples; // stop early if we reached target
        }
      }
    }
  }

  return samples;
}

std::vector<glm::vec3> sampleBoxVolume(const detail::Bounds &bounds,
                                       float spacing) {
  std::vector<glm::vec3> samples;

  glm::vec3 min = bounds.first;
  glm::vec3 max = bounds.second;
  glm::vec3 size = max - min;

  int pointsX, pointsY, pointsZ;
  glm::vec3 step;

  pointsX = std::max(1, int(std::floor(size.x / spacing)) + 1);
  pointsY = std::max(1, int(std::floor(size.y / spacing)) + 1);
  pointsZ = std::max(1, int(std::floor(size.z / spacing)) + 1);

  step = glm::vec3(spacing);

  for (int x = 0; x < pointsX; ++x) {
    for (int y = 0; y < pointsY; ++y) {
      for (int z = 0; z < pointsZ; ++z) {
        glm::vec3 point = min + glm::vec3(x, y, z) * step;
        samples.push_back(point);
      }
    }
  }

  return samples;
}

float point_spacing = 0.002f;
float sigmaP = 1.0f;
float sigmaN = 1.0f;
int gridResolution = 50;

int main() {

  // polyscope::options::autocenterStructures = true;
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

  drawPointCloud("cloud", cloud);

  const auto apss = APSS(cloud);

  const auto apssSDF = [&apss](const glm::vec3 &p) {
    const auto fitted = apss.fit(p, point_spacing * 5);

    return std::visit([&p](const auto &fit) { return distance(fit, p); },
                      fitted);
  };

  const auto pse_scale = 0.25f;
  auto bounds = detail::computeBoundingBox(cloud.positions, 2.f + pse_scale);

  // const auto extruded = sampleBoxSurface(bounds, 3000000);
  // const auto extruded = extrude(cloud, scale * 1.1).positions;
  const auto extruded = sampleBoxVolume(bounds, 30000);
  // drawPointCloud("extruded", extruded);

  auto dilated_points = extruded;
  auto dilated_normals = std::vector<PointCloud::Normal>(dilated_points.size());
  // now we want to project the extruded points down onto the dilation
  auto start = std::chrono::high_resolution_clock::now();
  auto end = start;
  for (auto i = 0; i < dilated_points.size(); i++) {

    if (i % 100000 == 0) {
      end = std::chrono::high_resolution_clock::now();
      std::cout << "Progress = "
                << i / static_cast<float>(dilated_points.size())
                << ", Duration = "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                         start)
                       .count()
                << " ms" << std::endl;
      start = std::chrono::high_resolution_clock::now();
    }

    // now we iteratively project
    auto [p, n] = structuring_elements::project_iterative<
        structuring_elements::Operation::Dilation>(
        apss, dilated_points[i], structuring_elements::sdf::sphere, pse_scale);

    dilated_points[i] = p;
    dilated_normals[i] = n;
  }
  auto dilatedPointCloud = PointCloud(dilated_points, dilated_normals);
  // dilatedPointCloud = poissonDiskSubsample(dilatedPointCloud, 0.1f);

  drawPointCloud("dilated", dilatedPointCloud);
  std::cout << "Projection done" << std::endl;

  auto dilatedPSS = APSS(dilatedPointCloud);

  const auto dilatedSDF = [&dilatedPSS, h = 0.5f](const auto &p) {
    return dilatedPSS.evaluate_surface(p, h);
  };

  addVolumeGrid(bounds, dilatedSDF, gridResolution);
  std::cout << "Done" << std::endl;

  polyscope::state::userCallback = [&]() {
    if (!polyscope::haveSelection()) {
      return;
    }
    auto sel = polyscope::getSelection();

    ImGui::Text("Selected point <%llu>", sel.localIndex);

    if (ImGui::Button("Do action")) {

      std::cout << sel.position.x << ", " << sel.position.y << ", "
                << sel.position.z << std::endl;

      auto fitted = structuring_elements::fit(
          apss, sel.position, structuring_elements::sdf::sphere, pse_scale);

      polyscope::PointCloud *psCloud =
          polyscope::registerPointCloud("selection", std::vector{fitted.c});

      auto quant = psCloud->addVectorQuantity(
          "distance", std::vector{sel.position - fitted.c});
      quant->setEnabled(true);
      quant->setVectorLengthScale(glm::distance(fitted.c, sel.position), false);
      psCloud->setPointRadius(0.02);

      // psCloud->setTransparency(0.2);
    }
  };

  polyscope::show();
}
