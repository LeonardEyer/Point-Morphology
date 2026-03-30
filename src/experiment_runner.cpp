#include "APSS.hpp"
#include "Draw.hpp"
#include "ExperimentConfig.hpp"
#include "MarkdownGallery.hpp"
#include "Morphology.hpp"
#include "PointCloud.hpp"
#include "PointStructuringElement.hpp"
#include "Resample.hpp"
#include "Subsample.hpp"
#include "Utils.hpp"

#include <chrono>
#include <filesystem>
#include <glm/glm.hpp>
#include <iostream>
#include <polyscope/point_cloud.h>
#include <polyscope/polyscope.h>
#include <polyscope/screenshot.h>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using SDF = structuring_elements::PointStructuringElement::SDF;

SDF getSDF(const std::string &shape) {
  if (shape == "sphere")
    return structuring_elements::sdf::sphere;
  if (shape == "cube")
    return structuring_elements::sdf::cube;
  if (shape == "torus")
    return structuring_elements::sdf::torus;
  throw std::runtime_error("Unknown structuring element shape: " + shape);
}

template <structuring_elements::Operation op>
gallery::ExperimentResult runExperiment(const ExperimentConfig &config,
                                        const fs::path &output_dir) {
  const auto &p = config.params;

  std::cout << "\n=== Running experiment: " << config.name
            << " ===" << std::endl;
  auto start = std::chrono::steady_clock::now();

  // Load point cloud
  auto cloud = fromPLY(config.input);
  size_t input_count = cloud.positions.size();
  std::cout << "  Loaded " << input_count << " points from " << config.input
            << std::endl;

  // Rescale to bounding box of 100
  {
    auto bounds = detail::computeBoundingBox(cloud.positions, 1);
    auto scaling = glm::distance(bounds.second, bounds.first);
    cloud.scale(100.0f / scaling);
  }

  auto sdf = getSDF(config.structuring_element);
  auto bounds = detail::computeBoundingBox(cloud.positions, p.pse_scale * 1.5f);
  auto original_apss = APSS(cloud, p.pss_scale);

  float radius = 2.5f * p.sigmaP;

  const auto makePointCentroid = [&](const auto &pt, const auto &) {
    const auto c =
        structuring_elements::fit(original_apss, pt, sdf, p.pse_scale).c;
    return util::concat(pt / p.sigmaP, c / p.pse_scale);
  };

  const auto makePointNormal = [&](const auto &pt, const auto &n) {
    return util::concat(pt / p.sigmaP, n / p.sigmaN);
  };

  cloud = subsample(cloud, radius * 1 / 5, makePointNormal);

  // Run morphological operation
  auto morphology_pointcloud =
      morphology::morph<op>(original_apss, bounds, sdf, p.sigmaP, p.pse_scale);

  auto morphology_subsampled =
      subsample(morphology_pointcloud, radius, makePointCentroid);

  const auto se_project = [&](const auto &pt) {
    return structuring_elements::project_iterative<op>(original_apss, pt, sdf,
                                                       p.pse_scale, 100, 1e-4f);
  };

  auto morphology_resampled =
      resample(morphology_subsampled, radius, makePointCentroid, se_project,
               p.resampling_iterations);

  size_t output_count = morphology_resampled.positions.size();

  // Visualize and screenshot
  polyscope::removeAllStructures();
  polyscope::view::setWindowSize(1024, 1024);

  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;

  draw::drawPointCloud("input", cloud)->setEnabled(false);
  draw::drawPointCloud("result", morphology_resampled);

  // Set up a reasonable camera view
  std::string json =
      R"({"farClipRatio":20.0,"fov":45.0,"nearClipRatio":0.005,"projectionMode":"Perspective","viewMat":[0.826793074607849,2.56113708019257e-09,0.562506377696991,0.0,-0.114914402365685,0.978909909725189,0.168905660510063,0.0,-0.550643444061279,-0.204290300607681,0.809355676174164,-121.243576049805,0.0,0.0,0.0,1.0],"windowHeight":1018,"windowWidth":1024})";

  polyscope::view::setCameraFromJson(json, false);

  auto screenshot_name = config.name + ".png";
  auto screenshot_path = output_dir / screenshot_name;
  polyscope::screenshot(screenshot_path.string(), false);

  auto end = std::chrono::steady_clock::now();
  double duration = std::chrono::duration<double>(end - start).count();

  std::cout << "  Completed in " << duration << "s"
            << " (" << output_count << " output points)" << std::endl;

  return gallery::ExperimentResult{
      .config = config,
      .screenshot_path = screenshot_name,
      .input_point_count = input_count,
      .output_point_count = output_count,
      .duration_seconds = duration,
  };
}

// Run a single morphological operation (dilation or erosion) and return the
// post-processed result.
template <structuring_elements::Operation op>
PointCloud runMorphOp(PointCloud &cloud, const ExperimentConfig &config,
                      const SDF &sdf) {
  const auto &p = config.params;

  auto bounds = detail::computeBoundingBox(cloud.positions, p.pse_scale * 1.5f);
  auto apss = APSS(cloud, p.pss_scale);

  auto morphology_pointcloud =
      morphology::morph<op>(apss, bounds, sdf, p.sigmaP, p.pse_scale);

  float radius = 2.5f * p.sigmaP;

  const auto makePointCentroid = [&](const auto &pt, const auto &) {
    const auto c = structuring_elements::fit(apss, pt, sdf, p.pse_scale).c;
    return util::concat(pt / p.sigmaP, c / p.pse_scale);
  };

  auto morphology_subsampled =
      subsample(morphology_pointcloud, radius, makePointCentroid);

  const auto se_project = [&](const auto &pt) {
    return structuring_elements::project_iterative<op>(apss, pt, sdf,
                                                       p.pse_scale, 100, 1e-4f);
  };

  return resample(morphology_subsampled, radius, makePointCentroid, se_project,
                  p.resampling_iterations);
}

gallery::ExperimentResult runExperiment(const ExperimentConfig &config,
                                        const fs::path &output_dir) {
  const auto &p = config.params;

  std::cout << "\n=== Running experiment: " << config.name
            << " ===" << std::endl;
  auto start = std::chrono::steady_clock::now();

  // Load point cloud
  auto cloud = fromPLY(config.input);
  size_t input_count = cloud.positions.size();
  std::cout << "  Loaded " << input_count << " points from " << config.input
            << std::endl;

  // Rescale to bounding box of 100
  {
    auto bounds = detail::computeBoundingBox(cloud.positions, 1);
    auto scaling = glm::distance(bounds.second, bounds.first);
    cloud.scale(100.0f / scaling);
  }

  const auto makePointNormal = [&](const auto &pt, const auto &n) {
    return util::concat(pt / (p.sigmaP * 1 / 5), n / p.sigmaN);
  };

  cloud = subsample(cloud, 2.5 * p.sigmaP * (1 / 5), makePointNormal);

  auto sdf = getSDF(config.structuring_element);

  PointCloud result_cloud;

  switch (config.operation) {
  case PipelineOperation::Dilation: {
    result_cloud =
        runMorphOp<structuring_elements::Dilation>(cloud, config, sdf);
    break;
  }
  case PipelineOperation::Erosion: {
    result_cloud =
        runMorphOp<structuring_elements::Erosion>(cloud, config, sdf);
    break;
  }
  case PipelineOperation::Opening: {
    // Opening = erosion followed by dilation
    std::cout << "  [Opening] Step 1/2: Erosion" << std::endl;
    auto eroded = runMorphOp<structuring_elements::Erosion>(cloud, config, sdf);
    std::cout << "  [Opening] Step 2/2: Dilation" << std::endl;
    result_cloud =
        runMorphOp<structuring_elements::Dilation>(eroded, config, sdf);
    break;
  }
  case PipelineOperation::Closing: {
    // Closing = dilation followed by erosion
    std::cout << "  [Closing] Step 1/2: Dilation" << std::endl;
    auto dilated =
        runMorphOp<structuring_elements::Dilation>(cloud, config, sdf);
    std::cout << "  [Closing] Step 2/2: Erosion" << std::endl;
    result_cloud =
        runMorphOp<structuring_elements::Erosion>(dilated, config, sdf);
    break;
  }
  }

  size_t output_count = result_cloud.positions.size();

  // Visualize and screenshot
  polyscope::removeAllStructures();
  polyscope::view::setWindowSize(1024, 1024);

  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;

  // draw::drawPointCloud("input", cloud)->setEnabled(false);
  // draw::drawPointCloud("result", result_cloud);

  auto apss = APSS(result_cloud, p.pss_scale);
  auto bounds = detail::computeBoundingBox(result_cloud.positions, 1);
  draw::addVolumeGridByResolution(
      "result", bounds,
      [&apss](const auto &p) { return apss.evaluate_surface(p); },
      p.grid_resolution);

  // Set up a reasonable camera view
  static const std::string json =
      R"({"farClipRatio":20.0,"fov":45.0,"nearClipRatio":0.005,"projectionMode":"Perspective","viewMat":[0.826793074607849,2.56113708019257e-09,0.562506377696991,0.0,-0.114914402365685,0.978909909725189,0.168905660510063,0.0,-0.550643444061279,-0.204290300607681,0.809355676174164,-121.243576049805,0.0,0.0,0.0,1.0],"windowHeight":1018,"windowWidth":1024})";

  polyscope::view::setCameraFromJson(json, false);

  auto screenshot_name = config.name + ".png";
  auto screenshot_path = output_dir / screenshot_name;
  polyscope::screenshot(screenshot_path.string(), false);

  auto end = std::chrono::steady_clock::now();
  double duration = std::chrono::duration<double>(end - start).count();

  std::cout << "  Completed in " << duration << "s"
            << " (" << output_count << " output points)" << std::endl;

  return gallery::ExperimentResult{
      .config = config,
      .screenshot_path = screenshot_name,
      .input_point_count = input_count,
      .output_point_count = output_count,
      .duration_seconds = duration,
  };
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <experiment.yaml>" << std::endl;
    return 1;
  }

  fs::path config_path = argv[1];
  auto pipeline = yaml::parse(config_path);

  std::cout << "Loaded " << pipeline.experiments.size() << " experiments from "
            << config_path << std::endl;

  // Resolve output directory relative to config file location
  auto base_dir = config_path.parent_path();
  auto output_dir = base_dir / pipeline.output_dir;
  fs::create_directories(output_dir);

  polyscope::init();

  std::vector<gallery::ExperimentResult> results;

  for (const auto &experiment : pipeline.experiments) {
    auto result = runExperiment(experiment, output_dir);
    results.push_back(std::move(result));
  }

  // Generate Markdown gallery
  gallery::generateMarkdown(output_dir, results);

  std::cout << "\nAll " << results.size() << " experiments complete."
            << std::endl;
  std::cout << "Results in: " << output_dir << std::endl;
  std::cout << "Gallery: " << (output_dir / "gallery.md") << std::endl;

  return 0;
}
