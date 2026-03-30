#include "APSS.hpp"
#include "Draw.hpp"
#include "Embedding.hpp"
#include "FeatureSample.hpp"
#include "Morphology.hpp"
#include "PointCloud.hpp"
#include "PointStructuringElement.hpp"
#include "Resample.hpp"
#include "Subsample.hpp"
#include "Utils.hpp"

#include <cassert>
#include <cmath>
#include <functional>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/trigonometric.hpp>
#include <imgui.h>
#include <iostream>
#include <nanoflann.hpp>
#include <polyscope/affine_remapper.h>
#include <polyscope/implicit_helpers.h>
#include <polyscope/options.h>
#include <polyscope/pick.h>
#include <polyscope/point_cloud.h>
#include <polyscope/point_cloud_vector_quantity.h>
#include <polyscope/polyscope.h>
#include <polyscope/screenshot.h>
#include <polyscope/slice_plane.h>
#include <polyscope/types.h>
#include <polyscope/utilities.h>
#include <polyscope/view.h>
#include <polyscope/volume_grid.h>
#include <stdexcept>

#include <utility>
#include <vector>

inline std::ostream &operator<<(std::ostream &os, const glm::mat3 &m) {
  os << "[\n";
  for (int row = 0; row < 3; ++row) {
    os << "  ";
    for (int col = 0; col < 3; ++col) {
      // Note the [column][row] indexing for column-major storage
      os << (col == 0 ? "" : ", ") << m[col][row];
    }
    os << "\n";
  }
  os << "]";
  return os;
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

int gridResolution = 100;

float pse_scale = 2.5f;
float pss_scale = 2.0f;
float sigmaP = std::min(pss_scale, pse_scale) / 2;
float radius = 2.5 * sigmaP;
float sigmaN = 0.75;
float &sigmaC = pse_scale;
int resampling_iterations = 10;
float mean_shift_search_radius = pse_scale;

bool pss_use_spacing = false;
bool sampling_compute_pointnormal_embedding = false;

static const char *pse_shapes[]{"Sphere", "Cube", "Torus", "Pointcloud"};
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
                           const auto &project, const auto &p, const auto &n,
                           bool showEmbedding) {

  auto neighbours = cloud_sampled.tree->neighboursInRadius(p, radius * sigmaP);

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
    auto nCloud = draw::drawPointCloud("neighbours", neighbours_points,
                                       polyscope::PointRenderMode::Sphere);

    auto quant = nCloud->addVectorQuantity("normals", neighbours_normals);

    // quant->setEnabled(true);
    auto importance =
        nCloud->addScalarQuantity("importance", neighbours_importance)
            ->setEnabled(true);
    // nCloud->setPointRadiusQuantity("importance");
    nCloud->setPointRadius(0.002);

    if (showEmbedding) {

      auto embeddingCloud =
          draw::drawPointCloud("embedding", neighbours_embed_tail,
                               polyscope::PointRenderMode::Sphere);

      std::vector<glm::vec3> neighbours_connections;

      for (auto i = 0; i < neighbours.size(); i++) {
        auto dist = neighbours_points[i] - neighbours_embed_tail[i];
        neighbours_connections.emplace_back(dist);
      }

      auto quant2 = embeddingCloud->addVectorQuantity(
          "connect", neighbours_connections, polyscope::VectorType::AMBIENT);
      // quant2->setVectorLengthScale(pse_scale, false);
    }
  }

  auto s_grad = importance_grad(x, neighbours_embedding, kernelResult.K,
                                kernelResult.k, kernelResult.active);
  auto s_grad_p = util::to_glm(s_grad.head(3));

  {
    auto pCloud = draw::drawPointCloud("point", std::vector{p},
                                       polyscope::PointRenderMode::Sphere);
    auto quant = pCloud->addVectorQuantity("s grad", std::vector{s_grad_p});
    quant->setVectorLengthScale(glm::length(s_grad_p), false);
    quant->setEnabled(true);
  }

  {
    auto start = p - 0.5f * s_grad_p;

    auto [pp1, nn1] = project(start);

    auto pCloud = draw::drawPointCloud("update", std::vector{start, pp1},
                                       std::vector{n, nn1},
                                       polyscope::PointRenderMode::Sphere);
  }
}

void iterative_projection_debug_view(const auto &pos, auto &projection,
                                     const auto &grad_distance) {
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec3> path;
  path.push_back({0, 0, 0});
  for (auto i = 0; i < resampling_iterations; i++) {

    const auto result = projection(pos, i);
    positions.push_back(result.first);
    normals.push_back(result.second);

    if (i > 0) {
      path.push_back(positions[i - 1] - positions[i]);
    }
  }

  std::vector<glm::vec3> gradients;
  std::vector<float> distances;
  for (const auto &p : positions) {
    const auto [grad, dist] = grad_distance(p);
    gradients.push_back(grad);
    distances.push_back(dist);
  }

  auto pc = draw::drawPointCloud("iterative projection",
                                 PointCloud(positions, normals),
                                 polyscope::PointRenderMode::Sphere);

  pc->addVectorQuantity("path", path, polyscope::VectorType::AMBIENT);
  pc->addVectorQuantity("gradient", gradients);
  pc->addScalarQuantity("distance", distances);
}

int main(int argc, char **argv) {

  if (argc < 3) {
    std::cout << "Usage: point-morphology -noff <path> OR -ply <path>"
              << std::endl;
    return 1;
  }
  polyscope::init();

  std::string flag = argv[1];
  std::string path = argv[2];
  PointCloud cloud; // Assuming your point cloud object type

  if (flag == "-noff") {
    cloud = readNOFF(path);
  } else if (flag == "-ply") {
    cloud = fromPLY(path);
  } else {
    std::cerr << "Unknown flag: " << flag << std::endl;
    return 1;
  }

  {
    polyscope::view::setWindowSize(1024, 1024);
    std::string json =
        R"({"farClipRatio":20.0,"fov":45.0,"nearClipRatio":0.005,"projectionMode":"Perspective","viewMat":[0.826793074607849,2.56113708019257e-09,0.562506377696991,0.0,-0.114914402365685,0.978909909725189,0.168905660510063,0.0,-0.550643444061279,-0.204290300607681,0.809355676174164,-121.243576049805,0.0,0.0,0.0,1.0],"windowHeight":1018,"windowWidth":1024})";

    polyscope::view::setCameraFromJson(json, false);
    polyscope::options::groundPlaneMode =
        polyscope::GroundPlaneMode::ShadowOnly;

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
  }

  // auto cloud = readNOFF("./resources/cone.txt");
  // auto cloud = readNOFF("./resources/cube.txt");
  //  auto cloud = readNOFF("./resources/thin-sheet.txt");
  //  auto cloud = fromPLY("./resources/hand2.ply");
  //  auto cloud = fromPLY("./resources/cap.ply");
  // auto cloud = fromPLY("./resources/armadillo.ply");
  //   auto cloud = fromPLY("./resources/bunny.ply");
  {
    // Rescale to have bounding box of 100
    auto bounds = detail::computeBoundingBox(cloud.positions, 0);
    auto scaling = glm::distance(bounds.second, bounds.first);
    cloud.scale(100 / scaling);
  }

  auto cloud_sampled =
      std::move(cloud); // subsample(cloud, radius, makePointNormal);
  auto subsampled = draw::drawPointCloud("subsampled", cloud_sampled);

  auto bounds =
      detail::computeBoundingBox(cloud_sampled.positions, pse_scale * 1.5);

  // Compute bounding box of the original cloud
  auto small_bounds = detail::computeBoundingBox(cloud_sampled.positions, 1);
  glm::vec3 minB = bounds.first;
  glm::vec3 maxB = bounds.second;
  glm::vec3 center = (minB + maxB) * 0.5f;
  glm::vec3 diff = maxB - minB;
  float scale = 1.0f / glm::compMax(diff); // largest axis -> 1

  const auto small_sdf = [=,
                          pss = APSS(cloud_sampled, pss_scale)](const auto &x) {
    // Scale query point back to original coordinates
    glm::vec3 original_p = x / scale + center;
    return pss.evaluate_surface(original_p);
  };

  polyscope::state::userCallback = [&]() {
    ImGui::InputFloat("radius", &radius);

    if (ImGui::InputFloat("PSE Scale", &pse_scale)) {
      bounds =
          detail::computeBoundingBox(cloud_sampled.positions, pse_scale * 1.5);
    }

    ImGui::InputFloat("PSS Kernel Support", &pss_scale);
    ImGui::InputFloat("sigmaP", &sigmaP);
    ImGui::InputFloat("sigmaN", &sigmaN);
    ImGui::InputFloat("sigmaC", &sigmaC);
    ImGui::InputInt("Grid resolution", &gridResolution);
    ImGui::InputFloat("Mean shift neighbours", &mean_shift_search_radius);
    ImGui::Checkbox("PSS use sigmaP", &pss_use_spacing);
    ImGui::Checkbox("Compute sampling using point and normal embedding",
                    &sampling_compute_pointnormal_embedding);

    if (ImGui::Button("Show subsampling")) {
      cloud_sampled = subsample(cloud, radius * sigmaP, makePointNormal);
      subsampled = draw::drawPointCloud("subsampled", cloud_sampled);
      log_point_cloud_stats(cloud_sampled);
    }
    ImGui::InputInt("Resampling iterations", &resampling_iterations);

    if (ImGui::Button("Resample")) {
      cloud_sampled = resample(
          cloud_sampled, radius * sigmaP, makePointNormal,
          [apss = APSS(cloud_sampled, pss_scale)](const auto &p) {
            return project_iterative(apss, p);
          },
          resampling_iterations);

      subsampled = draw::drawPointCloud("subsampled", cloud_sampled);
      log_point_cloud_stats(cloud_sampled);
    }

    ImGui::Combo("PSE shape", &selected_pse, pse_shapes, 4);
    ImGui::Combo("Morphological operation", &selected_morphology_op,
                 morphology_operations, 2);

    using namespace structuring_elements;
    const auto se_op = std::array{Operation::Dilation,
                                  Operation::Erosion}[selected_morphology_op];
    const auto sdf = std::array<std::function<float(const glm::vec3 &)>, 4>{
        sdf::sphere, sdf::roundcube, sdf::torus,
        [&](const glm::vec3 &p) { return small_sdf(p); }}[selected_pse];

    const auto morph = [&] {
      if (se_op == Operation::Dilation) {
        return morphology::morph<Dilation>;
      } else {
        return morphology::morph<Erosion>;
      }
    }();

    const auto se_project_iterative = [&] {
      if (se_op == Operation::Dilation) {
        return structuring_elements::project_iterative<Dilation>;

      } else {
        return structuring_elements::project_iterative<Erosion>;
      }
    }();
    auto original_apss = APSS(cloud_sampled, pss_scale);
    const auto morphological_iterative_projection = [&](const auto &p) {
      return se_project_iterative(original_apss, p, sdf, pse_scale, 100, 1e-4f);
    };

    if (ImGui::Button("Dense sampling")) {
      const auto denseSampling = morphology::detail::sampleBoxVolume(
          bounds, sigmaP, [&](const auto &sample) {
            auto dist = original_apss.evaluate_surface(sample);

            if (se_op == Erosion) {
              dist += pse_scale; // Sample close to dilation surface
            } else if (se_op == Dilation) {
              dist -= pse_scale;
            }

            return std::abs(dist) > 2 * sigmaP;
          });

      draw::drawPointCloud("dense sampling", denseSampling)->setEnabled(false);
    }

    const auto variational_dilation = [&](const auto &p) {
      const auto pse_distance =
          structuring_elements::fit(original_apss, p, sdf, pse_scale)
              .distance(p);
      const auto apss_distance = original_apss.evaluate_surface(p);

      return std::min(pse_distance, apss_distance);
    };

    const auto variational_erosion = [&](const auto &p) {
      const auto pse_distance =
          structuring_elements::fit(original_apss, p, sdf, pse_scale)
              .distance(p);
      const auto apss_distance = original_apss.evaluate_surface(p);

      return std::max(-pse_distance, apss_distance);
    };

    const auto variational_morphology = [&](const auto &p) {
      if (se_op == structuring_elements::Erosion) {
        return variational_erosion(p);
      }
      return variational_dilation(p);
    };

    if (ImGui::Button("Morphology (Variational)")) {

      draw::addVolumeGridByResolution("Morphology (Variational)", bounds,
                                variational_morphology, gridResolution);
    }

    if (ImGui::Button("Morphology (Sampled)")) {

      const auto morphology_pointcloud =
          morph(original_apss, bounds, sdf, sigmaP, pse_scale);

      const auto makePointCentroid = [sigmaC = sigmaC, sdf = sdf,
                                      &pss = original_apss](const auto &p,
                                                            const auto &n) {
        const auto c = structuring_elements::fit(pss, p, sdf, sigmaC).c;
        return util::concat(p / sigmaP, c / sigmaC);
      };

      auto morphology_subsampled =
          subsample(morphology_pointcloud, radius * sigmaP, makePointCentroid);

      auto morphology_resampled = resample(
          morphology_subsampled, radius * sigmaP, makePointCentroid,
          [&](const auto &p) {
            return se_project_iterative(original_apss, p, sdf, pse_scale, 100,
                                        1e-4f);
          },
          resampling_iterations);

      const auto morphology_edges = subsample(
          edge_sample(morphology_subsampled,
                      morphological_iterative_projection),
          radius * sigmaP,
          [sigmaP = sigmaP](const PointCloud::Position &p,
                            const PointCloud::Normal &n) -> Feature6D {
            return Feature6D(p.x, p.y, p.z, 0, 0, 0) / sigmaP;
          }); // do not compare normals

      draw::drawPointCloud("0: Morphology (initial)", morphology_pointcloud)
          ->setEnabled(false);

      draw::drawPointCloud("1: Morphology (subsampled)", morphology_subsampled)
          ->setEnabled(false);
      auto pc_resampled = draw::drawPointCloud("2: Morphology (resampled)",
                                               morphology_resampled);
      pc_resampled->setEnabled(false);

      if (sampling_compute_pointnormal_embedding) {
        auto morphology_subsampled_2 =
            subsample(morphology_pointcloud, radius * sigmaP, makePointNormal);
        auto morphology_resampled_2 = resample(
            morphology_subsampled_2, radius * sigmaP, makePointNormal,
            [&](const auto &p) {
              return se_project_iterative(original_apss, p, sdf, pse_scale, 100,
                                          1e-4f);
            },
            resampling_iterations);

        draw::drawPointCloud("1.5: Morphology(subsampled 2)",
                             morphology_subsampled_2)
            ->setEnabled(false);

        draw::drawPointCloud("2.5: Morphology(resampled 2)",
                             morphology_resampled_2)
            ->setEnabled(false);
      }

      // Prepare containers
      std::vector<glm::vec3> X, Y, Z;

      for (const auto &p : morphology_resampled.positions) {
        auto fit =
            structuring_elements::fit(original_apss, p, sdf, pse_scale, false);
        glm::mat3 J = fit.c_grad.value();

        // Extract Rows: Each represents the gradient of a center component
        // wrt
        // x

        X.push_back(glm::column(J, 0));
        Y.push_back(glm::column(J, 1));
        Z.push_back(glm::column(J, 2));
      }

      // Add to Polyscope
      pc_resampled->addVectorQuantity("Jacobian Col 0 (dc / dx)", X);
      pc_resampled->addVectorQuantity("Jacobian Col 1 (dc / dy)", Y);
      pc_resampled->addVectorQuantity("Jacobian Col 2 (dc / dz)", Z);
      draw::drawPointCloud("3: Morphology (edges)", morphology_edges)
          ->setEnabled(false);

      const auto morphology_pc_final = std::move(morphology_resampled);
      draw::drawPointCloud("4: Morphology (final)", morphology_pc_final);

      const auto morphology_apss = APSS(morphology_pc_final, pss_scale);
      draw::addVolumeGridByResolution(
          "Morphology (PSS)", bounds,
          [&morphology_apss, &pss_scale = pss_scale](const auto &p) {
            return morphology_apss.evaluate_surface(p);
          },
          gridResolution);
    }

    if (ImGui::Button("Feature detection")) {
      const auto edge_sampling = subsample(
          edge_sample(cloud_sampled, morphological_iterative_projection),
          radius * sigmaP, makePointNormal);

      draw::drawPointCloud("Edge sampling", edge_sampling);
    }

    if (ImGui::Button("Show gradients")) {
      auto pc = polyscope::getPointCloud("subsampled");

      std::vector<glm::vec3> gradients;
      for (const auto &p : cloud_sampled.positions) {
        gradients.push_back(std::visit(
            [&p](const auto &fitvariant) { return gradient(fitvariant, p); },
            original_apss.fit(p)));
      }
      pc->addVectorQuantity("gradients", gradients)->setEnabled(true);
    }

    if (ImGui::Button("Project all iteratively")) {

      std::vector<glm::vec3> positions;
      std::vector<glm::vec3> normals;

      for (const auto &p : cloud_sampled.positions) {
        auto [p2, n2] = project_iterative(original_apss, p);
        positions.push_back(p2);
        normals.push_back(n2);
      }
      draw::drawPointCloud("projected", PointCloud(positions, normals));
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

    if (ImGui::Button("Save pointcloud")) {
      if (sel.structureType == "Point Cloud") {
        auto pc = polyscope::getPointCloud(sel.structureName);
        const auto [positions, normals] = get_polyscope_data(pc);

        writeNOFF("./resources/" + sel.structureName + ".txt",
                  PointCloud(positions, normals));
      }
    }

    const auto [p, n] = [sel, &get_polyscope_data] {
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

    // if (ImGui::Button("Morphological: Project iteratively")) {
    //   const auto pos = PointCloud::Position(sel.position);
    //   const auto [new_p, new_n] =
    //       structuring_elements::project_iterative<se_op>(
    //           original_apss, pos, sdf, pss_scale, pse_scale);

    //   const auto pse_orig_fit =
    //       structuring_elements::fit(original_apss, pos, sdf, pse_scale);

    //   const auto pc0 = drawPointCloud(
    //       "Original Fitted center",
    //       PointCloud({pse_orig_fit.c}, {pse_orig_fit.gradient(pos)}),
    //       polyscope::PointRenderMode::Sphere);

    //   const auto pse_fit =
    //       structuring_elements::fit(original_apss, new_p, sdf, pse_scale);

    //   const auto pc = drawPointCloud("Iterative projection result",
    //                                  PointCloud({new_p}, {new_n}),
    //                                  polyscope::PointRenderMode::Sphere);

    //   const auto pc2 = drawPointCloud(
    //       "Fitted center", PointCloud({pse_fit.c},
    //       {pse_fit.gradient(new_p)}), polyscope::PointRenderMode::Sphere);

    //   pc->setPointRadius(0.002);
    //   pc2->setPointRadius(0.002);

    //   std::cout << "project_iterative: distance = " << glm::distance(new_p,
    //   pos)
    //             << std::endl;
    // }

    if (ImGui::Button("Project iteratively")) {
      const auto projection = [&original_apss](const auto &p, auto iter) {
        return project_iterative(original_apss, p, iter);
      };
      const auto grad_distance = [&original_apss](const auto &p) {
        const auto fit = original_apss.fit(p);
        return std::visit(
            [&p](const auto &fitvariant) {
              return std::make_pair(gradient(fitvariant, p),
                                    distance(fitvariant, p));
            },
            fit);
      };
      const auto pos = PointCloud::Position(sel.position);
      iterative_projection_debug_view(pos, projection, grad_distance);
    }
    if (ImGui::Button("Project iteratively (Morphological)")) {

      const auto projection = [&](const auto &p, auto iter) {
        return se_project_iterative(original_apss, p, sdf, pse_scale, iter,
                                    1e-4f);
      };
      const auto variational_morphology = [&](const auto &p) {
        const auto pse_distance =
            structuring_elements::fit(original_apss, p, sdf, pse_scale)
                .distance(p);
        const auto apss_distance = original_apss.evaluate_surface(p, pss_scale);

        if (se_op == structuring_elements::Erosion) {
          return std::max(-pse_distance, apss_distance);
        }

        return std::min(pse_distance, apss_distance);
      };

      const auto grad_distance = [&](const auto &p) {
        const auto pse_fit =
            structuring_elements::fit(original_apss, p, sdf, pse_scale);
        return std::make_pair(pse_fit.gradient(p), pse_fit.distance(p));
      };
      const auto pos = PointCloud::Position(sel.position);
      iterative_projection_debug_view(pos, projection, grad_distance);
    }

    // Reconstruct selected pointcloud
    if (sel.structureType != "Point Cloud") {
      return;
    }

    auto pc = polyscope::getPointCloud(sel.structureName);
    const auto [positions, normals] = get_polyscope_data(pc);
    const auto fetched_point_cloud = PointCloud(positions, normals);

    if (ImGui::Button("Importance debug view")) {

      importance_debug_view(
          fetched_point_cloud, makePointNormal,
          [apss = APSS(fetched_point_cloud, pss_scale)](const auto &p) {
            return project_iterative(apss, p);
          },
          p, n, false);
    }

    if (ImGui::Button("Show curvature")) {

      std::vector<float> curvature;
      for (auto i = 0; i < positions.size(); i++) {
        curvature.emplace_back(curvature_estimate(fetched_point_cloud, i));
      }
      pc->addScalarQuantity("curvature", curvature)->setEnabled(true);
    }

    if (ImGui::Button("(Morphological) Importance debug view")) {

      const auto makePointCentroid = [sigmaC = sigmaC, sdf = sdf,
                                      &pss = original_apss](const auto &p,
                                                            const auto &) {
        const auto c = structuring_elements::fit(pss, p, sdf, sigmaC).c;
        return util::concat(p / sigmaP, c / sigmaC);
      };

      importance_debug_view(
          fetched_point_cloud, makePointCentroid,
          [&](const auto &p) {
            return se_project_iterative(original_apss, p, sdf, pse_scale, 100,
                                        1e-4f);
          },
          p, n, true);
    }

    if (ImGui::Button("Show PSE")) {

      auto PSE_bounds = ::detail::Bounds(p - glm::vec3(1, 1, 1) * pse_scale,
                                         p + glm::vec3(1, 1, 1) * pse_scale);
      draw::addVolumeGridByResolution(
          "PSE", PSE_bounds,
          [&](const auto &x) {
            return structuring_elements::PointStructuringElement{pse_scale, p,
                                                                 sdf}
                .distance(x);
          },
          gridResolution);
    }

    if (ImGui::Button("Show PSS")) {

      const auto pss_sdf = [pss = APSS(fetched_point_cloud, pss_scale),
                            &pss_scale = pss_scale](const auto &p) {
        return pss.evaluate_surface(p);
      };

      if (pss_use_spacing) {
        auto max = 101;
        auto resolution = std::floor(max / sigmaP);

        draw::addVolumeGridByResolution("PSS", bounds, pss_sdf, resolution);
      } else {
        draw::addVolumeGridByResolution("PSS", bounds, pss_sdf, gridResolution);
      }
    }

    if (ImGui::Button("Show centroids")) {

      std::vector<glm::vec3> centroids;
      for (const auto &p : fetched_point_cloud.positions) {
        centroids.emplace_back(
            structuring_elements::fit(original_apss, p, sdf, sigmaC).c);
      }
      draw::drawPointCloud("centroids", centroids);
    }

    if (ImGui::Button("Mean shift")) {
      // We aim to minimize (9)

      const auto neighbours = [&, pse_scale = mean_shift_search_radius]() {
        std::vector<PointCloud::Position> neighbours;
        auto searchRadius = pse_scale;
        auto result =
            original_apss.pointCloud.tree->neighboursInRadius(p, searchRadius);
        while (result.empty()) {
          searchRadius *= 1.1;
          result = original_apss.pointCloud.tree->neighboursInRadius(
              p, searchRadius);
        }

        for (const auto &[idx, _] : result) {
          neighbours.emplace_back(original_apss.pointCloud.positions[idx]);
        }
        return neighbours;
      }();

      auto ppc =
          draw::drawPointCloud("p", {p}, polyscope::PointRenderMode::Sphere);

      auto npc = draw::drawPointCloud("knn", neighbours,
                                      polyscope::PointRenderMode::Sphere);

      // 2. initialize mean shift with n (usually 2) meaningful points
      // {c_j^0} := {closest points in PI under PSE distance}

      auto distances = std::vector<float>{};
      std::vector<std::pair<float, glm::vec3>> distances_and_points;

      for (const auto &xi : neighbours) {
        auto pse =
            structuring_elements::PointStructuringElement{pse_scale, xi, sdf};
        distances_and_points.emplace_back(pse.distance(p), xi);
        distances.emplace_back(distances_and_points.back().first);
      }

      npc->addScalarQuantity("distances", distances)->setEnabled(true);

      // we are looking at the 2 most meaningful points
      auto nth = 2;
      std::ranges::nth_element(distances_and_points,
                               distances_and_points.begin() + nth - 1,
                               [](const auto &lhs, const auto &rhs) {
                                 return lhs.first < rhs.first;
                               });

      auto cjs = std::vector<glm::vec3>{};
      for (const auto &[_, point] : distances_and_points) {
        cjs.emplace_back(point);
        if (cjs.size() == nth) {
          break;
        }
      }
      draw::drawPointCloud("candidates", cjs, polyscope::PointRenderMode::Quad);

      const auto [converged_candidates, converged_grads] =
          structuring_elements::mean_shift(p, cjs, neighbours, sdf, pse_scale);

      auto distances_converged = std::vector<float>{};
      std::vector<glm::vec3> gradX, gradY, gradZ;

      distances_converged.push_back(
          structuring_elements::PointStructuringElement{
              pse_scale, converged_candidates[0], sdf}
              .distance(p));

      auto J = converged_grads[0];

      std::cout << "J = " << J << std::endl;

      gradX.push_back(glm::normalize(glm::column(J, 0)));
      gradY.push_back(glm::normalize(glm::column(J, 1)));
      gradZ.push_back(glm::normalize(glm::column(J, 2)));

      auto convpc = draw::drawPointCloud("converged candidates",
                                         {converged_candidates[0]},
                                         polyscope::PointRenderMode::Sphere);
      convpc->addScalarQuantity("distance", distances_converged)
          ->setEnabled(false);
      convpc->addVectorQuantity("gradX", gradX);
      convpc->addVectorQuantity("gradY", gradY);
      convpc->addVectorQuantity("gradZ", gradZ);

      auto pse = structuring_elements::PointStructuringElement{
          pse_scale, converged_candidates[0], sdf, converged_grads[0]};

      auto grad_B = pse.gradient(p);

      // Erosion means flipping the pse normal
      if (se_op == structuring_elements::Erosion) {
        grad_B *= -1;
      }
      auto J_c = pse.c_grad.value();
      static constexpr glm::mat3 Identity(1.0f);

      // This term ensures that even if grad_B is [1,0,0],
      // it is transformed by the relationship between the point and the fit.
      auto normal = glm::normalize(grad_B * (Identity - J_c));

      ppc->addVectorQuantity("normal", std::vector{normal});
      ppc->addVectorQuantity(
          "gradient", std::vector{glm::normalize(
                          util::gradient(p, variational_morphology, sigmaP))});
    }
  };

  polyscope::screenshot();
  polyscope::show();
}
