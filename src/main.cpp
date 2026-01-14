#include "APSS.hpp"
#include "FeatureSample.hpp"
#include "InverseIterative.hpp"
#include "PointCloud.hpp"
#include "PointStructuringElement.hpp"
#include "Resample.hpp"
#include "Subsample.hpp"
#include "Utils.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/trigonometric.hpp"
#include "polyscope/slice_plane.h"

#include <fstream>
#include <imgui.h>
#include <nanoflann.hpp>
#include <polyscope/implicit_helpers.h>
#include <polyscope/pick.h>
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
  psGrid->setEdgeWidth(0);

  polyscope::VolumeGridNodeScalarQuantity *qNode =
      psGrid->addNodeScalarQuantity("sdf node", values);
  qNode->setEnabled(true);
  qNode->setIsolinesEnabled(true);
  qNode->setMapRange({-0.1, 0.1f});

  qNode->setGridcubeVizEnabled(true);
  qNode->setIsosurfaceLevel(0.0);
  // qNode->setIsosurfaceVizEnabled(true);
}

auto drawPointCloud(std::string name, const PointCloud &p) {
  auto cloud = polyscope::registerPointCloud(name, p.positions);
  cloud->addVectorQuantity("normals", p.normals);
  cloud->setPointRadius(0.002);
  cloud->setPointRenderMode(polyscope::PointRenderMode::Quad);

  return cloud;
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
int gridResolution = 50;

float pse_scale = 5.f;
float pss_scale = 1.0f;
float radius = 2.5;
float sigmaP = std::min(radius, pse_scale) / 2;
float sigmaN = 0.75;
int resampling_iterations = 10;

struct Kernel {
  Eigen::MatrixXd K; // inverse kernel matrix
  Eigen::VectorXd k;
};

util::Feature6D makePointNormal(const PointCloud::Position &position,
                                const PointCloud::Normal &normal) {
  return util::concat(position / sigmaP, normal / sigmaN);
}

util::Features6D
makePointNormals(const std::vector<PointCloud::Position> &positions,
                 const std::vector<PointCloud::Normal> &normals) {
  auto result = std::vector<util::Feature6D>();

  for (auto i = 0; i < positions.size(); i++) {
    result.push_back(makePointNormal(positions[i], normals[i]));
  }

  return result;
}

int main() {
  polyscope::init();

  auto plane1 = polyscope::addSceneSlicePlane(true);
  plane1->setDrawWidget(false);
  plane1->setDrawPlane(false);

  auto plane2 = polyscope::addSceneSlicePlane(true);
  plane2->setDrawWidget(false);
  plane2->setDrawPlane(false);
  plane2->setTransform(
      glm::rotate(glm::mat4(1.0f), glm::radians(90.f), glm ::vec3(0, 1, 0)));

  auto cloud = readNOFF("./resources/cube.txt");
  {
    // Rescale to have bounding box of 100
    auto bounds = detail::computeBoundingBox(cloud.positions, 1);
    auto scaling = glm::distance(bounds.second, bounds.first);
    std::cout << "scaling = " << scaling << std::endl;
    cloud.scale(100 / scaling);
  }

  auto cloud_sampled = subsample(cloud, radius, makePointNormal);

  auto subsampled = drawPointCloud("subsampled", cloud_sampled);

  const auto apss = APSS(cloud_sampled);
  const auto makeSDF = [&apss](const auto radius) {
    return [&apss, radius](const auto &p) {
      return std::visit([&p](const auto &fit) { return distance(fit, p); },
                        apss.fit(p, radius));
    };
  };
  // const auto cloud_resampled =
  //     resample(cloud_sampled, radius, makePointNormal,
  //     resampling_iterations);

  // drawPointCloud("resampled", cloud_resampled);

  const auto bounds = detail::computeBoundingBox(cloud_sampled.positions, 1.5f);

  polyscope::state::userCallback = [&]() {
    ImGui::InputFloat("radius", &radius);
    ImGui::InputFloat("PSE Scale", &pse_scale);
    ImGui::InputFloat("PSS Kernel Support", &pss_scale);

    ImGui::InputFloat("sigmaP", &sigmaP);
    ImGui::InputFloat("sigmaN", &sigmaN);

    ImGui::InputInt("Grid resolution", &gridResolution);

    if (ImGui::Button("Show PSS")) {
      addVolumeGrid(bounds, makeSDF(pss_scale), gridResolution);
    }

    if (ImGui::Button("Show subsampling")) {
      cloud_sampled = subsample(cloud, radius, makePointNormal);
      subsampled = drawPointCloud("subsampled", cloud_sampled);
      log_point_cloud_stats(cloud_sampled);
    }
    ImGui::InputInt("Resampling iterations", &resampling_iterations);

    if (ImGui::Button("Resample")) {
      const auto cloud_resampled = resample(
          cloud_sampled, radius, makePointNormal, resampling_iterations);

      drawPointCloud("resampled", cloud_resampled);
      log_point_cloud_stats(cloud_resampled);
    }

    if (ImGui::Button("Morphology (Dilation)")) {
      static constexpr auto sdf = structuring_elements::sdf::cube;

      const auto denseSampling = sampleBoxVolume(bounds, sigmaP);

      std::cout << "Dense sampling count = " << denseSampling.size()
                << std::endl;
      auto dilated_points = denseSampling;
      auto dilated_normals =
          std::vector<PointCloud::Normal>(dilated_points.size());
      // now we want to project the extruded points down onto the dilation
      auto start = std::chrono::high_resolution_clock::now();
      auto end = start;
      for (auto i = 0; i < dilated_points.size(); i++) {

        if (i != 0 && i % 100000 == 0) {
          end = std::chrono::high_resolution_clock::now();
          std::cout << "Progress = "
                    << i / static_cast<float>(dilated_points.size())
                    << ", Duration = "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(
                           end - start)
                           .count()
                    << " ms" << std::endl;
          start = std::chrono::high_resolution_clock::now();
        }

        // now we iteratively project
        auto [p, n] = structuring_elements::project_iterative<
            structuring_elements::Dilation>(apss, dilated_points[i], sdf,
                                            pse_scale);

        dilated_points[i] = p;
        dilated_normals[i] = n;
      }

      auto dilatedPointCloud = PointCloud(dilated_points, dilated_normals);

      auto dilatedSubsampled =
          subsample(dilatedPointCloud, radius, makePointNormal);

      const auto makePointCentroid = [sigmaC = pse_scale, sdf = sdf,
                                      &pss = apss](const auto &p,
                                                   const auto &) {
        const auto c = structuring_elements::fit(pss, p, sdf, sigmaC).c;
        return util::concat(p / sigmaP, c / sigmaC);
      };

      // auto dilatedResampled = resample(
      //     dilatedPointCloud, radius, makePointCentroid,
      //     resampling_iterations);

      const auto dilatedEdges =
          subsample(edge_sample(dilatedSubsampled), radius, makePointNormal);

      drawPointCloud("dilated", dilatedPointCloud);
      drawPointCloud("dilated subsampled", dilatedSubsampled);
      //    drawPointCloud("dilated resampled", dilatedResampled);
      drawPointCloud("dilated edges", dilatedEdges);
    }

    if (ImGui::Button("Feature detection")) {
      const auto edge_sampling =
          subsample(edge_sample(cloud_sampled), radius, makePointNormal);

      drawPointCloud("Edge sampling", edge_sampling);
    }

    if (!polyscope::haveSelection()) {
      return;
    }
    const auto sel = polyscope::getSelection();
    const auto &p = cloud_sampled.positions[sel.localIndex];
    const auto &n = cloud_sampled.normals[sel.localIndex];

    if (ImGui::Button("Compute APSS fit")) {
      const auto pos = PointCloud::Position(sel.position);
      const auto fitresult = apss.fit(pos, pss_scale);
      std::visit(
          [](const auto &variant) {
            using T = std::decay_t<decltype(variant)>;
            if constexpr (std::is_same_v<T, SphereFitResult>) {
              std::cout << "Sphere" << std::endl;
            } else if constexpr (std::is_same_v<T, PlaneFitResult>) {
              std::cout << "Plane" << std::endl;
            }
          },
          fitresult);
    }

    if (ImGui::Button("Show curvature")) {

      if (curvature_estimate(cloud_sampled, sel.localIndex) < 0.75) {
        const auto A_qem = quadric(cloud_sampled, sel.localIndex);
        const auto projected = projectToFeature(p, n, A_qem);

        if (projected) {
          drawPointCloud("projection", std::vector{p, projected->first},
                         polyscope::PointRenderMode::Sphere);
        }
      }

      std::vector<float> curvature;
      for (auto i = 0; i < cloud_sampled.positions.size(); i++) {
        curvature.emplace_back(curvature_estimate(cloud_sampled, i));
      }

      subsampled->addScalarQuantity("curvature", curvature)->setEnabled(true);
    }

    if (ImGui::Button("Show nearest in radius")) {

      auto neighbours = cloud_sampled.tree->neighboursInRadius(p, radius);

      if (neighbours.empty()) {
        std::cout << "No neighbours" << std::endl;
      }

      auto neighbours_points = std::vector<PointCloud::Position>{};
      auto neighbours_normals = std::vector<PointCloud::Normal>{};
      auto neighbours_importance = std::vector<float>{};
      for (const auto [index, _] : neighbours) {
        neighbours_points.push_back(cloud_sampled.positions[index]);
        neighbours_normals.push_back(cloud_sampled.normals[index]);

        auto importance =
            rbfKernel(makePointNormal(p, n),
                      makePointNormal(cloud_sampled.positions[index],
                                      cloud_sampled.normals[index]));

        neighbours_importance.push_back(importance);
      }

      auto pointNormal = makePointNormal(p, n);
      auto neighbours6D =
          makePointNormals(neighbours_points, neighbours_normals);

      auto kernelResult =
          takeInverseIterative(pointNormal, neighbours6D, rbfKernel, -1);

      auto s = kernelResult.s;

      {
        auto nCloud = drawPointCloud("neighbours", neighbours_points,
                                     polyscope::PointRenderMode::Sphere);

        auto quant = nCloud->addVectorQuantity("normals", neighbours_normals);

        // quant->setEnabled(true);
        auto importance =
            nCloud->addScalarQuantity("importance", neighbours_importance)
                ->setEnabled(true);
        nCloud->setPointRadiusQuantity("importance");
        nCloud->setPointRadius(0.002);
      }

      auto s_grad = importance_grad(pointNormal, neighbours6D, kernelResult.K,
                                    kernelResult.k, kernelResult.active);
      auto s_grad_p = util::to_glm(s_grad.head(3));
      // std::cout << "s_grad =\n" << s_grad << std::endl;
      {
        auto pCloud = drawPointCloud("point", std::vector{p},
                                     polyscope::PointRenderMode::Sphere);
        auto quant = pCloud->addVectorQuantity("s grad", std::vector{s_grad_p});
        quant->setVectorLengthScale(glm::length(s_grad_p), false);
        quant->setEnabled(true);
      }

      {
        auto start = p - 0.5f * s_grad_p;

        auto [pp1, _] = project_iterative(apss, start, radius);

        auto pCloud = drawPointCloud("update", std::vector{start, pp1},
                                     polyscope::PointRenderMode::Sphere);
      }
    }
  };

  polyscope::show();
}
