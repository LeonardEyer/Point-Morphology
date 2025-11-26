#include "APSS.hpp"
#include "InverseIterative.hpp"
#include "PointCloud.hpp"
#include "PointStructuringElement.hpp"
#include "Resample.hpp"
#include "Subsample.hpp"
#include "imgui.h"
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

auto drawPointCloud(
    std::string name, const std::vector<PointCloud::Position> &p,
    polyscope::PointRenderMode mode = polyscope::PointRenderMode::Quad) {
  auto *cloud = polyscope::registerPointCloud(name, p);
  cloud->setPointRadius(0.002);
  cloud->setPointRenderMode(mode);

  return cloud;
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

  const auto max_spacing = maximum_point_spacing(cloud);
  const auto avg_spacing = average_point_spacing(cloud);
  const auto min_spacing = minimum_point_spacing(cloud);

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
float sigmaP = .25f;
float sigmaN = .5f;
int gridResolution = 50;

using V6f = Eigen::Vector<float, 6>;
using PointNormal = V6f;

struct Kernel {
  Eigen::MatrixXd K; // inverse kernel matrix
  Eigen::VectorXd k;
};

PointNormal make6D(const PointCloud::Position &position,
                   const PointCloud::Normal &normal) {
  return concat(position / sigmaP, normal / sigmaN);
}

std::vector<PointNormal>
make6D(const std::vector<PointCloud::Position> &positions,
       const std::vector<PointCloud::Normal> &normals) {
  auto result = std::vector<PointNormal>();

  for (auto i = 0; i < positions.size(); i++) {
    result.push_back(make6D(positions[i], normals[i]));
  }

  return result;
}

template <typename KernelFunc>
Kernel buildInverseKernelMatrix(const PointNormal &x,
                                const std::vector<PointNormal> &neighbours,
                                const KernelFunc &kernelFunc) {

  const auto N = neighbours.size();
  std::cout << "N = " << N << std::endl;
  Eigen::MatrixXd K(N, N);
  Eigen::VectorXd k(N);

  for (auto i = 0; i < N; i++) {
    k(i) = kernelFunc(x, neighbours[i]);
  }

  for (auto i = 0; i < N; i++) {
    for (auto j = 0; j < N; j++) {
      K(i, j) = kernelFunc(neighbours[i], neighbours[j]);
    }
  }

  std::cout << "invert" << std::endl;

  // check if PSD matrix

  std::cout << "k = " << k << std::endl;
  // std::cout << "K = " << K << std::endl;

  // K = K.inverse();
  // std::cout << "K^-1 = " << K << std::endl;

  return {K.inverse(), k};
};

float importance(Kernel kernelResult) {
  const auto &[K, k] = kernelResult;

  return 1.0 - k.transpose() * K * k;
};

PointNormal importance_grad(const PointNormal &x,
                            const std::vector<PointNormal> &neighbours,
                            Kernel kernelResult, float sigma = 1.0f) {

  auto [K, k] = kernelResult;

  PointNormal sgrad = PointNormal::Zero();
  for (auto i = 0; i < neighbours.size(); i++) {
    for (auto j = 0; j < neighbours.size(); j++) {

      const auto &x_i = neighbours[i];
      const auto &x_j = neighbours[j];

      sgrad += (2 * x - x_i - x_j) * k(i) * k(j) * K(i, j);
    }
  }
  return (-2.0f / sigma * sigma) * sgrad;
}

PointNormal importance_grad(const PointNormal &x,
                            const std::vector<PointNormal> &neighbours,
                            KernelInverseResult kernelResult,
                            float sigma = 1.0f) {

  auto [_, K, k, active] = kernelResult;

  auto neighboursFiltered = std::vector<PointNormal>();
  for (auto i = 0; i < neighbours.size(); i++) {
    if (active[i]) {
      neighboursFiltered.push_back(neighbours[i]);
    }
  }

  PointNormal sgrad = PointNormal::Zero();
  for (auto i = 0; i < neighboursFiltered.size(); i++) {
    for (auto j = 0; j < neighboursFiltered.size(); j++) {

      const auto &x_i = neighboursFiltered[i];
      const auto &x_j = neighboursFiltered[j];

      sgrad += (2 * x - x_i - x_j) * k(i) * k(j) * K(i, j);
    }
  }
  return (-2.0f / sigma * sigma) * sgrad;
}

glm::vec3 toglm(Eigen::Vector3f x) { return {x.x(), x.y(), x.z()}; }

int main() {

  // polyscope::options::autocenterStructures = true;
  polyscope::view::windowWidth = 1024;
  polyscope::view::windowHeight = 1024;
  polyscope::init();

  const auto cloud = readNOFF("./resources/cube-sampled.txt");

  // // resample(handSubsampled, gaussianStd * gaussianStd, 1u);

  drawPointCloud("cloud", cloud);

  const auto apss = APSS(cloud);

  const auto apssSDF = [&apss](const glm::vec3 &p) {
    const auto fitted = apss.fit(p, point_spacing * 5);

    return std::visit([&p](const auto &fit) { return distance(fit, p); },
                      fitted);
  };

  constexpr auto pse_scale = 0.25f;
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

    if (i != 0 && i % 100000 == 0) {
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
  std::cout << "Projection done" << std::endl;
  // drawPointCloud("dilated", dilatedPointCloud);

  dilatedPointCloud = poissonDiskSubsample(dilatedPointCloud, 0.05f);
  std::cout << "subsampling done" << std::endl;
  drawPointCloud("dilated subsampled", dilatedPointCloud);

  // auto dilatedPointCloud2 = resample(dilatedPointCloud, 0.5, 1.5, 1.25);
  // std::cout << "resampling done" << std::endl;
  // drawPointCloud("dilated resampled", dilatedPointCloud2);

  auto dilatedPSS = APSS(dilatedPointCloud);

  const auto dilatedSDF = [&dilatedPSS, h = 0.5f](const auto &p) {
    return dilatedPSS.evaluate_surface(p, h);
  };

  // addVolumeGrid(bounds, dilatedSDF, gridResolution);
  std::cout << "Done" << std::endl;

  float radius = 0.5;

  polyscope::state::userCallback = [&]() {
    if (!polyscope::haveSelection()) {
      return;
    }
    auto sel = polyscope::getSelection();

    ImGui::InputFloat("radius", &radius);
    ImGui::InputFloat("sigmaP", &sigmaP);
    ImGui::InputFloat("sigmaN", &sigmaN);

    if (ImGui::Button("Show PSE centroid")) {

      auto fitted = structuring_elements::fit(
          apss, sel.position, structuring_elements::sdf::sphere, pse_scale);

      polyscope::PointCloud *psCloud =
          polyscope::registerPointCloud("selection", std::vector{fitted.c});

      auto quant = psCloud->addVectorQuantity(
          "distance", std::vector{sel.position - fitted.c});

      quant->setEnabled(true);
      quant->setVectorLengthScale(glm::distance(fitted.c, sel.position), false);
      psCloud->setPointRadius(0.02);
    }
    if (ImGui::Button("Show nearest in radius")) {

      const auto &cloud = dilatedPointCloud;

      auto &p = cloud.positions[sel.localIndex];
      auto &n = cloud.normals[sel.localIndex];

      auto neighbours = cloud.tree->neighboursInRadius(p, radius);

      if (neighbours.empty()) {
        std::cout << "No neighbours" << std::endl;
      }

      auto neighbours_points = std::vector<PointCloud::Position>{};
      auto neighbours_normals = std::vector<PointCloud::Normal>{};
      auto neighbours_importance = std::vector<float>{};
      for (const auto [index, _] : neighbours) {
        neighbours_points.push_back(cloud.positions[index]);
        neighbours_normals.push_back(cloud.normals[index]);

        auto importance = kernel6D(
            make6D(p, n), make6D(cloud.positions[index], cloud.normals[index]));

        neighbours_importance.push_back(importance);
      }

      auto pointNormal = make6D(p, n);
      auto neighbours6D = make6D(neighbours_points, neighbours_normals);

      auto kernelResult = takeInverseIterative(pointNormal, neighbours6D);
      auto s2 = importance(Kernel{kernelResult.K, kernelResult.k});
      auto s = kernelResult.s;

      // std::cout << "s = " << s << std::endl;
      // std::cout << "s2 = " << 1.0 - s2 << std::endl;

      {
        auto nCloud = drawPointCloud("neighbours", neighbours_points,
                                     polyscope::PointRenderMode::Sphere);

        auto quant = nCloud->addVectorQuantity("normals", neighbours_normals);

        // quant->setEnabled(true);
        auto importance =
            nCloud->addScalarQuantity("importance", neighbours_importance)
                ->setEnabled(true);
        nCloud->setPointRadiusQuantity("importance");
        nCloud->setPointRadius(0.02);
      }

      auto s_grad =
          0.5f * importance_grad(pointNormal, neighbours6D, kernelResult);

      auto s_grad_p = glm::vec3(s_grad.x(), s_grad.y(), s_grad.z());
      {
        auto pCloud = drawPointCloud("point", std::vector{p},
                                     polyscope::PointRenderMode::Sphere);
        auto quant = pCloud->addVectorQuantity("s grad", std::vector{s_grad_p});
        quant->setVectorLengthScale(glm::length(s_grad_p), false);
        quant->setEnabled(true);
      }

      {

        auto [pp1, np1] =
            project_iterative(dilatedPSS, p + toglm(s_grad.head(3)), 0.5, 1);

        auto pCloud = drawPointCloud("update", std::vector{pp1},
                                     polyscope::PointRenderMode::Sphere);
      }
    }
  };

  polyscope::show();
}
