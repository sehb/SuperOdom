#pragma once

#ifndef SUPER_ODOMETRY_VISUAL_ODOMETRY_H
#define SUPER_ODOMETRY_VISUAL_ODOMETRY_H

#include <map>
#include <mutex>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <cv_bridge/cv_bridge.h>
#include <builtin_interfaces/msg/time.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "super_odometry/config/parameter.h"
#include "super_odometry/utils/Twist.h"

namespace super_odometry {

struct VisualOdometryConfig {
  std::string image_topic;
  std::string imu_odom_topic;
  std::string output_topic;
  std::string world_frame;
  std::string camera_frame;
  int max_features;
  int min_matches;
  int min_inliers;
  double fx;
  double fy;
  double cx;
  double cy;
  double max_time_diff_s;
  double max_visual_speed;
  double max_imu_speed_for_scale;
  double min_imu_scale;
  double max_imu_scale;
  double imu_rotation_weight;
};

class VisualOdometry : public rclcpp::Node {
public:
  explicit VisualOdometry(const rclcpp::NodeOptions &options);

  void initInterface();

private:
  void loadParameters();

  void imuOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr msg);

  void imageHandler(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  bool getImuOdomAtTime(double timestamp, Transformd &T_w_i) const;

  bool estimateRelativePose(const cv::Mat &prev_img, const cv::Mat &curr_img,
                            Eigen::Quaterniond &q_prev_curr,
                            Eigen::Vector3d &t_prev_curr_dir, int &inliers,
                            int &n_matches);

  void publishOdometry(double timestamp);

  static double secs(const builtin_interfaces::msg::Time &stamp);

private:
  VisualOdometryConfig config_{};

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_imu_odom_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_visual_odom_;

  mutable std::mutex mtx_;
  std::map<double, Transformd> imu_odom_buf_;

  cv::Ptr<cv::ORB> orb_;
  cv::BFMatcher matcher_{cv::NORM_HAMMING, false};

  cv::Mat prev_gray_;
  double prev_img_time_ = -1.0;
  bool has_prev_frame_ = false;
  bool has_pose_ = false;
  Transformd T_w_c_;
};

} // namespace super_odometry

#endif // SUPER_ODOMETRY_VISUAL_ODOMETRY_H
