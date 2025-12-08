#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <unordered_map>

using Point = Eigen::Vector3f;

template <typename T> struct VoxelGrid {
  using VoxelID = int;
  using Coordinate = std::array<int, 3>;

  struct CoordHash {
    size_t operator()(const std::array<int, 3> &c) const noexcept {
      size_t h = 0;
      for (int i = 0; i < 3; ++i)
        h ^= std::hash<int>{}(c[i]) + 0x9e3779b97f4a7c15ULL + (h << 6) +
             (h >> 2);
      return h;
    }
  };

  Point voxelsize;
  std::unordered_map<Coordinate, T, CoordHash> data;

  Coordinate to_coordinate(const Point &pos) {
    const auto floored =
        (pos.array() / voxelsize.array()).floor().template cast<int>();
    return {floored.x(), floored.y(), floored.z()};
  }

  void insert(const Point &pos, const T &date) {
    data[to_coordinate(pos)] = date;
  }

  const std::optional<T> get(const Point &pos) {
    const auto coord = to_coordinate(pos);

    if (!data.contains(coord)) {
      return std::nullopt;
    }

    return data[coord];
  }
};
