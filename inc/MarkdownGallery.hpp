#pragma once

#include "ExperimentConfig.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace gallery {

struct ExperimentResult {
  ExperimentConfig config;
  std::string screenshot_path; // relative to output_dir
  size_t input_point_count = 0;
  size_t output_point_count = 0;
  double duration_seconds = 0.0;
};

inline std::string operationStr(PipelineOperation op) {
  switch (op) {
  case PipelineOperation::Dilation: return "Dilation";
  case PipelineOperation::Erosion: return "Erosion";
  case PipelineOperation::Opening: return "Opening";
  case PipelineOperation::Closing: return "Closing";
  }
  return "Unknown";
}

inline void generateMarkdown(const std::filesystem::path &output_dir,
                             const std::vector<ExperimentResult> &results) {

  std::ostringstream md;

  md << "# Point Morphology Experiments\n\n";
  md << "Generated experiment results gallery.\n\n";

  for (const auto &r : results) {
    auto op = operationStr(r.config.operation);

    md << "---\n\n";
    md << "## " << r.config.name << "\n\n";
    md << "**Operation:** " << op << "\n\n";

    md << "<img src='" << r.screenshot_path <<"' alt='" << r.config.name <<"' width='400'/>\n\n";
    
    md << "| Parameter | Value |\n";
    md << "|-----------|-------|\n";
    md << "| Input | " << r.config.input << " |\n";
    md << "| SE Shape | " << r.config.structuring_element << " |\n";
    md << "| PSE Scale | " << r.config.params.pse_scale << " |\n";
    md << "| PSS Scale | " << r.config.params.pss_scale << " |\n";
    md << "| sigmaP | " << r.config.params.sigmaP << " |\n";
    md << "| sigmaN | " << r.config.params.sigmaN << " |\n";
    md << "| Resample Iters | " << r.config.params.resampling_iterations
       << " |\n";
    md << "| Grid Res | " << r.config.params.grid_resolution << " |\n";
    md << "\n";
    md << "**Results:** " << r.input_point_count << " pts in, "
       << r.output_point_count << " pts out";

    if (r.duration_seconds > 0) {
      md << " | " << std::fixed << std::setprecision(1) << r.duration_seconds
         << "s";
    }

    md << "\n\n";
  }

  auto gallery_path = output_dir / "gallery.md";
  std::ofstream out(gallery_path);
  if (!out.is_open())
    throw std::runtime_error("Cannot write gallery: " + gallery_path.string());
  out << md.str();
  std::cout << "Markdown gallery written to " << gallery_path << std::endl;
}

} // namespace gallery
