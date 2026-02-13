#pragma once

#include <polyscope/point_cloud.h>
#include <polyscope/polyscope.h>

#include "PointCloud.hpp"

inline auto drawPointCloud(
    std::string name, const PointCloud &p,
    polyscope::PointRenderMode mode = polyscope::PointRenderMode::Quad) {
  auto cloud = polyscope::registerPointCloud(name, p.positions);
  cloud->addVectorQuantity("normals", p.normals);
  cloud->setPointRadius(0.002);
  cloud->setPointRenderMode(mode);

  return cloud;
}

inline auto drawPointCloud(
    std::string name, const std::vector<PointCloud::Position> &p,
    const std::vector<PointCloud::Normal> &n,
    polyscope::PointRenderMode mode = polyscope::PointRenderMode::Quad) {
  auto cloud = polyscope::registerPointCloud(name, p);
  cloud->addVectorQuantity("normals", n);
  cloud->setPointRadius(0.002);
  cloud->setPointRenderMode(mode);

  return cloud;
}

inline auto drawPointCloud(
    std::string name, const std::vector<PointCloud::Position> &p,
    polyscope::PointRenderMode mode = polyscope::PointRenderMode::Quad) {
  auto *cloud = polyscope::registerPointCloud(name, p);
  cloud->setPointRadius(0.002);
  cloud->setPointRenderMode(mode);

  return cloud;
}
