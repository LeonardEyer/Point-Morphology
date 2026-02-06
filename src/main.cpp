#include "APSS.hpp"
#include "Embedding.hpp"
#include "FeatureSample.hpp"
#include "PointCloud.hpp"
#include "PointStructuringElement.hpp"
#include "Resample.hpp"
#include "Subsample.hpp"
#include "Utils.hpp"

#include <atomic>
#include <cmath>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <glm/trigonometric.hpp>
#include <imgui.h>
#include <iostream>
#include <nanoflann.hpp>
#include <polyscope/affine_remapper.h>
#include <polyscope/implicit_helpers.h>
#include <polyscope/pick.h>
#include <polyscope/point_cloud.h>
#include <polyscope/point_cloud_vector_quantity.h>
#include <polyscope/polyscope.h>
#include <polyscope/slice_plane.h>
#include <polyscope/types.h>
#include <polyscope/utilities.h>
#include <polyscope/volume_grid.h>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace detail {

template <typename SDFFunc>
void computeVolumeGrid(const glm::vec3 &minBound, const glm::vec3 &maxBound,
                       uint32_t dimX, uint32_t dimY, uint32_t dimZ,
                       float spacingX, float spacingY, float spacingZ,
                       const SDFFunc &sdf, std::vector<float> &values,
                       std::atomic<size_t> &voxelsDone) {

  unsigned numThreads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;

  auto worker = [&](size_t zStart, size_t zEnd) {
    for (size_t iz = zStart; iz < zEnd; ++iz) {
      for (size_t iy = 0; iy < dimY; ++iy) {
        for (size_t ix = 0; ix < dimX; ++ix) {
          size_t idx = iz * dimY * dimX + iy * dimX + ix;

          glm::vec3 p(minBound.x + ix * spacingX, minBound.y + iy * spacingY,
                      minBound.z + iz * spacingZ);

          // Clamp to max bounds
          p.x = std::min(p.x, maxBound.x);
          p.y = std::min(p.y, maxBound.y);
          p.z = std::min(p.z, maxBound.z);

          values[idx] = sdf(p);
          voxelsDone.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
  };

  size_t chunk = dimZ / numThreads;
  size_t zStart = 0;
  for (unsigned t = 0; t < numThreads; ++t) {
    size_t zEnd = (t == numThreads - 1) ? dimZ : zStart + chunk;
    threads.emplace_back(worker, zStart, zEnd);
    zStart = zEnd;
  }

  size_t totalVoxels = dimX * dimY * dimZ;
  while (voxelsDone < totalVoxels) {
    std::cout << "\rProgress: " << (100.0 * voxelsDone / totalVoxels) << "%   "
              << std::flush;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  for (auto &t : threads)
    t.join();
  std::cout << "\rProgress: 100%   \n";
  voxelsDone = 0;
}

} // namespace detail

template <typename SDFFunc>
void addVolumeGridByResolution(const std::string &name,
                               const detail::Bounds &bounds, const SDFFunc &sdf,
                               const size_t resolution = 50) {
  auto &[minBound, maxBound] = bounds;

  uint32_t dimX = resolution;
  uint32_t dimY = resolution;
  uint32_t dimZ = resolution;

  float spacingX = (maxBound.x - minBound.x) / (dimX - 1);
  float spacingY = (maxBound.y - minBound.y) / (dimY - 1);
  float spacingZ = (maxBound.z - minBound.z) / (dimZ - 1);

  std::vector<float> values(dimX * dimY * dimZ);
  static std::atomic<size_t> voxelsDone{0};

  detail::computeVolumeGrid(minBound, maxBound, dimX, dimY, dimZ, spacingX,
                            spacingY, spacingZ, sdf, values, voxelsDone);

  auto *psGrid = polyscope::registerVolumeGrid(name, {dimX, dimY, dimZ},
                                               minBound, maxBound);
  psGrid->setEdgeWidth(0);
  auto *qNode = psGrid->addNodeScalarQuantity("sdf node", values);
  qNode->setEnabled(true);
  qNode->setIsosurfaceLevel(0.0);
  qNode->setIsosurfaceVizEnabled(true);
  qNode->setGridcubeVizEnabled(false);
}

template <typename SDFFunc>
void addVolumeGridBySpacing(const std::string &name,
                            const detail::Bounds &bounds, const SDFFunc &sdf,
                            const float spacing = 0.1f) {

  throw std::runtime_error("Not working");
  auto &[minBound, maxBound] = bounds;

  // Compute voxel counts
  uint32_t dimX =
      static_cast<uint32_t>(std::ceil((maxBound.x - minBound.x) / spacing));
  uint32_t dimY =
      static_cast<uint32_t>(std::ceil((maxBound.y - minBound.y) / spacing));
  uint32_t dimZ =
      static_cast<uint32_t>(std::ceil((maxBound.z - minBound.z) / spacing));

  float spacingX = spacing;
  float spacingY = spacing;
  float spacingZ = spacing;

  std::vector<float> values(dimX * dimY * dimZ);
  static std::atomic<size_t> voxelsDone{0};

  detail::computeVolumeGrid(minBound, maxBound, dimX, dimY, dimZ, spacingX,
                            spacingY, spacingZ, sdf, values, voxelsDone);

  auto *psGrid = polyscope::registerVolumeGrid(name, {dimX, dimY, dimZ},
                                               minBound, maxBound);
  psGrid->setEdgeWidth(0);
  auto *qNode = psGrid->addNodeScalarQuantity("sdf node", values);
  qNode->setEnabled(true);
  qNode->setIsosurfaceLevel(0.0);
  qNode->setIsosurfaceVizEnabled(true);
  qNode->setGridcubeVizEnabled(false);
}

auto drawPointCloud(
    std::string name, const PointCloud &p,
    polyscope::PointRenderMode mode = polyscope::PointRenderMode::Quad) {
  auto cloud = polyscope::registerPointCloud(name, p.positions);
  cloud->addVectorQuantity("normals", p.normals);
  cloud->setPointRadius(0.002);
  cloud->setPointRenderMode(mode);

  return cloud;
}

auto drawPointCloud(
    std::string name, const std::vector<PointCloud::Position> &p,
    const std::vector<PointCloud::Normal> &n,
    polyscope::PointRenderMode mode = polyscope::PointRenderMode::Quad) {
  auto cloud = polyscope::registerPointCloud(name, p);
  cloud->addVectorQuantity("normals", n);
  cloud->setPointRadius(0.002);
  cloud->setPointRenderMode(mode);

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
int gridResolution = 100;

float pse_scale = 2.f;
float pss_scale = 1.0f;
float radius = 2.5;
float sigmaP = std::min(radius, pse_scale) / 2;
float sigmaN = 0.75;
float sigmaC = pse_scale;
int resampling_iterations = 10;

bool pss_use_spacing = false;

static const structuring_elements::Operation se_op =
    structuring_elements::Operation::Dilation;

static constexpr auto sdf = structuring_elements::sdf::cube;

static const char *pse_shapes[]{"Sphere", "Cube"};
int selected_pse = 0;

static const char *morphology_operations[]{"Dilation", "Erosion"};
int selected_morphology_op = 0;

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

void importance_debug_view(const auto &cloud_sampled, const auto &embedding,
                           const auto &project, const auto &p, const auto &n) {

  auto neighbours = cloud_sampled.tree->neighboursInRadius(p, radius);

  // remove oneself
  neighbours.erase(neighbours.begin());

  if (neighbours.empty()) {
    std::cout << "No neighbours" << std::endl;
  }

  auto neighbours_points = std::vector<PointCloud::Position>{};
  auto neighbours_normals = std::vector<PointCloud::Normal>{};
  auto neighbours_embedding = std::vector<Feature6D>{};
  auto neighbours_importance = std::vector<float>{};
  auto neighbours_embed_tail = std::vector<PointCloud::Position>{};

  for (const auto [index, _] : neighbours) {
    neighbours_points.push_back(cloud_sampled.positions[index]);
    neighbours_normals.push_back(cloud_sampled.normals[index]);
    neighbours_embedding.push_back(embedding(cloud_sampled.positions[index],
                                             cloud_sampled.normals[index]));
    neighbours_embed_tail.push_back(
        util::to_glm(neighbours_embedding.back().tail(3)) * sigmaC);

    auto importance = rbfKernel(embedding(p, n), neighbours_embedding.back());
    neighbours_importance.push_back(importance);
  }

  auto x = embedding(p, n);

  auto kernelResult =
      takeInverseIterative(x, neighbours_embedding, rbfKernel, -1);

  auto s = kernelResult.s;

  {
    auto nCloud = drawPointCloud("neighbours", neighbours_points,
                                 polyscope::PointRenderMode::Sphere);

    auto quant = nCloud->addVectorQuantity("normals", neighbours_normals);

    // quant->setEnabled(true);
    auto importance =
        nCloud->addScalarQuantity("importance", neighbours_importance)
            ->setEnabled(true);
    // nCloud->setPointRadiusQuantity("importance");
    nCloud->setPointRadius(0.002);

    // {

    //   auto embeddingCloud = drawPointCloud("embedding",
    //   neighbours_embed_tail,
    //                                        polyscope::PointRenderMode::Sphere);

    //   std::vector<glm::vec3> neighbours_connections;

    //   for (auto i = 0; i < neighbours.size(); i++) {
    //     auto dist = neighbours_points[i] - neighbours_embed_tail[i];
    //     neighbours_connections.emplace_back(dist);
    //   }

    //   auto quant2 = embeddingCloud->addVectorQuantity(
    //       "connect", neighbours_connections, polyscope::VectorType::AMBIENT);
    //   // quant2->setVectorLengthScale(pse_scale, false);
    // }
  }

  auto s_grad = importance_grad(x, neighbours_embedding, kernelResult.K,
                                kernelResult.k, kernelResult.active);
  auto s_grad_p = util::to_glm(s_grad.head(3));

  {
    auto pCloud = drawPointCloud("point", std::vector{p},
                                 polyscope::PointRenderMode::Sphere);
    auto quant = pCloud->addVectorQuantity("s grad", std::vector{s_grad_p});
    quant->setVectorLengthScale(glm::length(s_grad_p), false);
    quant->setEnabled(true);
  }

  {
    auto start = p - 0.5f * s_grad_p;

    auto [pp1, nn1] = project(start);

    auto pCloud =
        drawPointCloud("update", std::vector{start, pp1}, std::vector{n, nn1},
                       polyscope::PointRenderMode::Sphere);
  }
}

int main() {
  polyscope::init();

  auto plane1 = polyscope::addSceneSlicePlane(true);
  plane1->setDrawWidget(false);
  plane1->setDrawPlane(false);
  plane1->setActive(false);

  auto plane2 = polyscope::addSceneSlicePlane(true);
  plane2->setDrawWidget(false);
  plane2->setDrawPlane(false);
  plane2->setTransform(
      glm::rotate(glm::mat4(1.0f), glm::radians(90.f), glm ::vec3(0, 1, 0)));
  plane2->setActive(false);

  // auto cloud = readNOFF("./resources/cube.txt");

  auto cloud = fromPLY("./resources/hand2.ply");
  {
    // Rescale to have bounding box of 100
    auto bounds = detail::computeBoundingBox(cloud.positions, 1);
    auto scaling = glm::distance(bounds.second, bounds.first);
    std::cout << "scaling = " << scaling << std::endl;
    cloud.scale(100 / scaling);
  }

  auto cloud_sampled = subsample(cloud, radius, makePointNormal);

  auto subsampled = drawPointCloud("subsampled", cloud_sampled);

  const auto original_apss = APSS(cloud_sampled);
  const auto makeSDF = [&original_apss](const auto radius) {
    return [&original_apss, radius](const auto &p) {
      return std::visit([&p](const auto &fit) { return distance(fit, p); },
                        original_apss.fit(p, radius));
    };
  };

  const auto bounds = detail::computeBoundingBox(cloud_sampled.positions, 1.5f);

  polyscope::state::userCallback = [&]() {
    ImGui::InputFloat("radius", &radius);
    ImGui::InputFloat("PSE Scale", &pse_scale);
    ImGui::InputFloat("PSS Kernel Support", &pss_scale);
    ImGui::InputFloat("sigmaP", &sigmaP);
    ImGui::InputFloat("sigmaN", &sigmaN);
    ImGui::InputFloat("sigmaC", &sigmaC);
    ImGui::InputInt("Grid resolution", &gridResolution);
    ImGui::Checkbox("PSS use sigmaP", &pss_use_spacing);

    if (ImGui::Button("Show PSS")) {

      if (pss_use_spacing) {
        addVolumeGridBySpacing("PSS", bounds, makeSDF(pss_scale), sigmaP);
      } else {
        addVolumeGridByResolution("PSS", bounds, makeSDF(pss_scale),
                                  gridResolution);
      }
    }

    if (ImGui::Button("Show subsampling")) {
      cloud_sampled = subsample(cloud, radius, makePointNormal);
      subsampled = drawPointCloud("subsampled", cloud_sampled);
      log_point_cloud_stats(cloud_sampled);
    }
    ImGui::InputInt("Resampling iterations", &resampling_iterations);

    if (ImGui::Button("Resample")) {
      const auto cloud_resampled = resample(
          cloud_sampled, radius, makePointNormal,
          [apss = APSS(cloud_sampled)](const auto &p) {
            return project_iterative(apss, p, pss_scale);
          },
          resampling_iterations);

      drawPointCloud("resampled", cloud_resampled);
      log_point_cloud_stats(cloud_resampled);
    }

    // auto combo = ImGui::Combo("PSE shape", &selected_pse, pse_shapes, 2);

    if (ImGui::Button("Morphology (Dilation)")) {
      const auto denseSampling = sampleBoxVolume(bounds, sigmaP);

      // drawPointCloud("dense sampling", denseSampling);

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
        auto [p, n] = structuring_elements::project_iterative<se_op>(
            original_apss, dilated_points[i], sdf, pse_scale);

        dilated_points[i] = p;
        dilated_normals[i] = n;
      }

      auto dilatedPointCloud = PointCloud(dilated_points, dilated_normals);

      const auto makePointCentroid = [sigmaC = sigmaC, sdf = sdf,
                                      &pss = original_apss](const auto &p,
                                                            const auto &) {
        const auto c = structuring_elements::fit(pss, p, sdf, sigmaC).c;
        return util::concat(p / sigmaP, c / sigmaC);
      };

      auto dilatedSubsampled =
          subsample(dilatedPointCloud, radius, makePointCentroid);

      {
        addVolumeGridByResolution(
            "Dilated PSS", bounds,
            [original_apss](const auto &p) {
              const auto pse_distance =
                  structuring_elements::fit(original_apss, p, sdf, pse_scale)
                      .distance(p);
              const auto apss_distance =
                  original_apss.evaluate_surface(p, radius);
              return std::min(pse_distance, apss_distance);
            },
            gridResolution);
      }

      auto dilatedResampled = resample(
          dilatedSubsampled, radius, makePointCentroid,
          [&original_apss](const auto &p) {
            return structuring_elements::project_iterative<se_op>(
                original_apss, p, sdf, pse_scale);
          },
          resampling_iterations);

      // auto dilatedResampled = resample(
      //     dilatedSubsampled, radius, makePointNormal,
      //     [apss = APSS(dilatedSubsampled)](const auto &p) {
      //       return project_iterative(apss, p, pss_scale);
      //     },
      //     resampling_iterations);

      const auto dilatedEdges = subsample(
          edge_sample(dilatedSubsampled), radius,
          [sigmaP = sigmaP](const PointCloud::Position &p,
                            const PointCloud::Normal &n) -> Feature6D {
            return Feature6D(p.x, p.y, p.z, 0, 0, 0) / sigmaP;
          }); // do not compare normals

      // drawPointCloud("dilated", dilatedPointCloud);
      // drawPointCloud("dilated subsampled", dilatedSubsampled);
      drawPointCloud("dilated resampled", dilatedResampled);
      drawPointCloud("dilated edges", dilatedEdges);
    }

    if (ImGui::Button("Feature detection")) {
      const auto edge_sampling =
          subsample(edge_sample(cloud_sampled), radius, makePointNormal);

      drawPointCloud("Edge sampling", edge_sampling);
    }

    if (ImGui::Button("Show gradients")) {
      auto pc = polyscope::getPointCloud("subsampled");

      std::vector<glm::vec3> gradients;
      for (const auto &p : cloud_sampled.positions) {
        gradients.push_back(std::visit(
            [&p](const auto &fitvariant) { return gradient(fitvariant, p); },
            original_apss.fit(p, pss_scale)));
      }
      pc->addVectorQuantity("gradients", gradients)->setEnabled(true);
    }

    if (ImGui::Button("Project all iteratively")) {

      std::vector<glm::vec3> positions;
      std::vector<glm::vec3> normals;

      for (const auto &p : cloud_sampled.positions) {
        auto [p2, n2] = project_iterative(original_apss, p, pss_scale);
        positions.push_back(p2);
        normals.push_back(n2);
      }
      drawPointCloud("projected", PointCloud(positions, normals));
    }

    if (!polyscope::haveSelection()) {
      return;
    }

    const auto sel = polyscope::getSelection();

    const auto get_polyscope_data = [](polyscope::PointCloud *pc) {
      const auto positions = std::vector(pc->points.data);

      if (!pc->quantities.contains("normals")) {
        return std::make_pair(
            positions, std::vector<glm::vec3>(positions.size(), glm::vec3(0)));
      }

      auto *quant = dynamic_cast<polyscope::PointCloudVectorQuantity *>(
          pc->getQuantity("normals"));

      if (!quant) {
        throw std::runtime_error(
            "Quantity 'normals' is not a PointCloudVectorQuantity");
      }

      const auto normals = std::vector(quant->vectors.data);
      return std::make_pair(positions, normals);
    };

    const auto [p, n] = [sel, &get_polyscope_data]() {
      if (sel.structureType == "Point Cloud") {
        auto pc = polyscope::getPointCloud(sel.structureName);
        const auto [positions, normals] = get_polyscope_data(pc);

        return std::make_pair(positions[sel.localIndex],
                              normals[sel.localIndex]);
      } else {
        return std::make_pair(PointCloud::Position(sel.position),
                              PointCloud::Normal(1, 0, 0));
      }
    }();

    if (ImGui::Button("Morphological: Project iteratively")) {
      const auto pos = PointCloud::Position(sel.position);
      const auto [new_p, new_n] =
          structuring_elements::project_iterative<se_op>(original_apss, pos,
                                                         sdf, pse_scale);

      const auto pse_orig_fit =
          structuring_elements::fit(original_apss, pos, sdf, pse_scale);

      const auto pc0 = drawPointCloud(
          "Original Fitted center",
          PointCloud({pse_orig_fit.c}, {pse_orig_fit.gradient(pos)}),
          polyscope::PointRenderMode::Sphere);

      const auto pse_fit =
          structuring_elements::fit(original_apss, new_p, sdf, pse_scale);

      const auto pc = drawPointCloud("Iterative projection result",
                                     PointCloud({new_p}, {new_n}),
                                     polyscope::PointRenderMode::Sphere);

      const auto pc2 = drawPointCloud(
          "Fitted center", PointCloud({pse_fit.c}, {pse_fit.gradient(new_p)}),
          polyscope::PointRenderMode::Sphere);

      pc->setPointRadius(0.002);
      pc2->setPointRadius(0.002);

      std::cout << "project_iterative: distance = " << glm::distance(new_p, pos)
                << std::endl;
    }

    if (ImGui::Button("Project iteratively")) {
      const auto pos = PointCloud::Position(sel.position);

      std::vector<glm::vec3> positions;
      std::vector<glm::vec3> normals;
      std::vector<glm::vec3> path;
      path.push_back({0, 0, 0});
      for (auto i = 0; i < resampling_iterations; i++) {
        const auto result = project_iterative(original_apss, pos, pss_scale, i);
        positions.push_back(result.first);
        normals.push_back(result.second);

        if (i > 0) {
          path.push_back(positions[i - 1] - positions[i]);
        }
      }

      std::vector<glm::vec3> gradients;
      std::vector<float> distances;
      for (const auto &p : positions) {
        const auto fit = original_apss.fit(p, pss_scale);
        const auto [grad, dist] = std::visit(
            [&p](const auto &fitvariant) {
              return std::make_pair(gradient(fitvariant, p),
                                    distance(fitvariant, p));
            },
            fit);
        gradients.push_back(grad);
        distances.push_back(dist);
      }

      auto pc =
          drawPointCloud("iterative projection", PointCloud(positions, normals),
                         polyscope::PointRenderMode::Sphere);

      pc->addVectorQuantity("path", path, polyscope::VectorType::AMBIENT);
      pc->addVectorQuantity("gradient", gradients);
      pc->addScalarQuantity("distance", distances);
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

    if (ImGui::Button("Importance debug view")) {

      auto pc = polyscope::getPointCloud(sel.structureName);
      const auto [positions, normals] = get_polyscope_data(pc);
      const auto fetched_point_cloud = PointCloud(positions, normals);
      importance_debug_view(
          fetched_point_cloud, makePointNormal,
          [apss = APSS(fetched_point_cloud)](const auto &p) {
            return project_iterative(apss, p, radius);
          },
          p, n);
    }

    if (ImGui::Button(" (Morphological) Importance debug view")) {
      auto pc = polyscope::getPointCloud(sel.structureName);
      const auto [positions, normals] = get_polyscope_data(pc);

      const auto fetched_point_cloud = PointCloud(positions, normals);
      const auto makePointCentroid = [sigmaC = sigmaC, sdf = sdf,
                                      &pss = original_apss](const auto &p,
                                                            const auto &) {
        const auto c = structuring_elements::fit(pss, p, sdf, sigmaC).c;
        return util::concat(p / sigmaP, c / sigmaC);
      };

      importance_debug_view(
          fetched_point_cloud, makePointCentroid,
          [&original_apss](const auto &p) {
            return structuring_elements::project_iterative<se_op>(
                original_apss, p, sdf, pse_scale);
          },
          p, n);
    }
  };

  polyscope::show();
}
