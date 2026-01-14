#pragma once

#include "PointCloud.hpp"

#include <Eigen/Dense>

using Feature6D = Eigen::Vector<float, 6>;

template <typename F>
concept ImportanceEmbedding =
    requires(const PointCloud::Position &pos, const PointCloud::Normal &norm) {
      { std::declval<F>()(pos, norm) } -> std::same_as<Feature6D>;
    };
