#include "rclcpp/rclcpp.hpp"
#include "super_odometry/VisualOdometry/visualOdometry.h"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.arguments({"visual_odometry_node"});

  auto visual_odometry = std::make_shared<super_odometry::VisualOdometry>(options);
  visual_odometry->initInterface();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(visual_odometry);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
