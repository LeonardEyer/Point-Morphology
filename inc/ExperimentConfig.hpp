#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "PointStructuringElement.hpp"

// High-level pipeline operations, including composite ones.
// Opening = erosion then dilation; Closing = dilation then erosion.
enum class PipelineOperation { Dilation, Erosion, Opening, Closing };

struct ExperimentParams {
  float pse_scale = 2.5f;
  float pss_scale = 2.0f;
  float sigmaP = 1.25f;
  float sigmaN = 0.75f;
  int resampling_iterations = 10;
  int grid_resolution = 100;
};

struct ExperimentConfig {
  std::string name;
  std::string input;
  PipelineOperation operation = PipelineOperation::Dilation;
  std::string structuring_element = "sphere";
  ExperimentParams params;
};

struct PipelineConfig {
  std::string output_dir = "results";
  ExperimentParams defaults;
  std::vector<ExperimentConfig> experiments;
};

namespace yaml {

// Minimal YAML parser for the experiment config format.
// Supports the flat key-value and list-of-maps structure we need
// without requiring an external YAML library.

namespace detail {

inline std::string trim(const std::string &s) {
  auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos)
    return "";
  auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

inline int indentLevel(const std::string &line) {
  int count = 0;
  for (char c : line) {
    if (c == ' ')
      count++;
    else
      break;
  }
  return count;
}

inline std::pair<std::string, std::string> splitKeyValue(const std::string &line) {
  auto colon = line.find(':');
  if (colon == std::string::npos)
    return {"", ""};
  auto key = trim(line.substr(0, colon));
  auto value = trim(line.substr(colon + 1));
  return {key, value};
}

inline void applyParam(ExperimentParams &p, const std::string &key,
                        const std::string &value) {
  if (key == "pse_scale")
    p.pse_scale = std::stof(value);
  else if (key == "pss_scale")
    p.pss_scale = std::stof(value);
  else if (key == "sigmaP")
    p.sigmaP = std::stof(value);
  else if (key == "sigmaN")
    p.sigmaN = std::stof(value);
  else if (key == "resampling_iterations")
    p.resampling_iterations = std::stoi(value);
  else if (key == "grid_resolution")
    p.grid_resolution = std::stoi(value);
}

inline PipelineOperation parseOperation(const std::string &s) {
  if (s == "erosion")
    return PipelineOperation::Erosion;
  if (s == "opening")
    return PipelineOperation::Opening;
  if (s == "closing")
    return PipelineOperation::Closing;
  return PipelineOperation::Dilation;
}

} // namespace detail

inline PipelineConfig parse(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("Cannot open config: " + path.string());

  PipelineConfig config;
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line))
    lines.push_back(line);

  enum class Section { Root, Defaults, Experiments };
  Section section = Section::Root;
  ExperimentConfig *current_exp = nullptr;

  for (size_t i = 0; i < lines.size(); i++) {
    const auto &raw = lines[i];
    auto trimmed = detail::trim(raw);
    if (trimmed.empty() || trimmed[0] == '#')
      continue;

    int indent = detail::indentLevel(raw);
    auto [key, value] = detail::splitKeyValue(trimmed);

    // Handle list items (- key: value)
    bool is_list_item = false;
    if (trimmed.size() > 1 && trimmed[0] == '-' && trimmed[1] == ' ') {
      is_list_item = true;
      auto inner = detail::trim(trimmed.substr(2));
      auto [k2, v2] = detail::splitKeyValue(inner);
      key = k2;
      value = v2;
    }

    // Section detection at root level
    if (indent == 0 && !is_list_item) {
      if (key == "output_dir") {
        config.output_dir = value;
        section = Section::Root;
        continue;
      } else if (key == "defaults") {
        section = Section::Defaults;
        continue;
      } else if (key == "experiments") {
        section = Section::Experiments;
        continue;
      }
    }

    if (section == Section::Defaults) {
      detail::applyParam(config.defaults, key, value);
    } else if (section == Section::Experiments) {
      if (is_list_item) {
        // Start a new experiment, inheriting defaults
        config.experiments.emplace_back();
        current_exp = &config.experiments.back();
        current_exp->params = config.defaults;
      }
      if (current_exp) {
        if (key == "name")
          current_exp->name = value;
        else if (key == "input")
          current_exp->input = value;
        else if (key == "operation")
          current_exp->operation = detail::parseOperation(value);
        else if (key == "structuring_element")
          current_exp->structuring_element = value;
        else
          detail::applyParam(current_exp->params, key, value);
      }
    }
  }

  return config;
}

} // namespace yaml
