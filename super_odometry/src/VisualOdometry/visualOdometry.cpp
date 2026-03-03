#include "super_odometry/VisualOdometry/visualOdometry.h"

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>

namespace super_odometry {

VisualOdometry::VisualOdometry(const rclcpp::NodeOptions &options)
    : Node("visual_odometry_node", options) {}

void VisualOdometry::initInterface() {
  if (!readGlobalparam(shared_from_this())) {
    RCLCPP_ERROR(this->get_logger(),
                 "[VisualOdometry] Failed to read global parameters.");
    rclcpp::shutdown();
    return;
  }

  loadParameters();
  orb_ = cv::ORB::create(config_.max_features);

  sub_imu_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
      config_.imu_odom_topic, 100,
      std::bind(&VisualOdometry::imuOdometryHandler, this,
                std::placeholders::_1));

  sub_image_ = this->create_subscription<sensor_msgs::msg::Image>(
      config_.image_topic, 20,
      std::bind(&VisualOdometry::imageHandler, this, std::placeholders::_1));

  pub_visual_odom_ =
      this->create_publisher<nav_msgs::msg::Odometry>(config_.output_topic, 20);

  RCLCPP_INFO(this->get_logger(),
              "[VisualOdometry] Running. image_topic=%s imu_odom_topic=%s "
              "output_topic=%s",
              config_.image_topic.c_str(), config_.imu_odom_topic.c_str(),
              config_.output_topic.c_str());
}

void VisualOdometry::loadParameters() {
  this->declare_parameter<std::string>("visual_odometry_node.image_topic",
                                       "/camera/image_raw");
  this->declare_parameter<std::string>("visual_odometry_node.imu_odom_topic",
                                       ProjectName + "/state_estimation");
  this->declare_parameter<std::string>("visual_odometry_node.output_topic",
                                       ProjectName + "/visual_odometry");
  this->declare_parameter<std::string>("visual_odometry_node.world_frame",
                                       WORLD_FRAME);
  this->declare_parameter<std::string>("visual_odometry_node.camera_frame",
                                       SENSOR_FRAME);
  this->declare_parameter<int>("visual_odometry_node.max_features", 1000);
  this->declare_parameter<int>("visual_odometry_node.min_matches", 80);
  this->declare_parameter<int>("visual_odometry_node.min_inliers", 35);
  this->declare_parameter<double>("visual_odometry_node.fx", 525.0);
  this->declare_parameter<double>("visual_odometry_node.fy", 525.0);
  this->declare_parameter<double>("visual_odometry_node.cx", 319.5);
  this->declare_parameter<double>("visual_odometry_node.cy", 239.5);
  this->declare_parameter<double>("visual_odometry_node.max_time_diff_s", 0.05);
  this->declare_parameter<double>("visual_odometry_node.max_visual_speed", 15.0);
  this->declare_parameter<double>("visual_odometry_node.max_imu_speed_for_scale",
                                  15.0);
  this->declare_parameter<double>("visual_odometry_node.min_imu_scale", 0.01);
  this->declare_parameter<double>("visual_odometry_node.max_imu_scale", 1.5);
  this->declare_parameter<double>("visual_odometry_node.imu_rotation_weight",
                                  0.3);

  config_.image_topic =
      this->get_parameter("visual_odometry_node.image_topic").as_string();
  config_.imu_odom_topic =
      this->get_parameter("visual_odometry_node.imu_odom_topic").as_string();
  config_.output_topic =
      this->get_parameter("visual_odometry_node.output_topic").as_string();
  config_.world_frame =
      this->get_parameter("visual_odometry_node.world_frame").as_string();
  config_.camera_frame =
      this->get_parameter("visual_odometry_node.camera_frame").as_string();
  config_.max_features =
      this->get_parameter("visual_odometry_node.max_features").as_int();
  config_.min_matches =
      this->get_parameter("visual_odometry_node.min_matches").as_int();
  config_.min_inliers =
      this->get_parameter("visual_odometry_node.min_inliers").as_int();
  config_.fx = this->get_parameter("visual_odometry_node.fx").as_double();
  config_.fy = this->get_parameter("visual_odometry_node.fy").as_double();
  config_.cx = this->get_parameter("visual_odometry_node.cx").as_double();
  config_.cy = this->get_parameter("visual_odometry_node.cy").as_double();
  config_.max_time_diff_s =
      this->get_parameter("visual_odometry_node.max_time_diff_s").as_double();
  config_.max_visual_speed =
      this->get_parameter("visual_odometry_node.max_visual_speed").as_double();
  config_.max_imu_speed_for_scale =
      this->get_parameter("visual_odometry_node.max_imu_speed_for_scale")
          .as_double();
  config_.min_imu_scale =
      this->get_parameter("visual_odometry_node.min_imu_scale").as_double();
  config_.max_imu_scale =
      this->get_parameter("visual_odometry_node.max_imu_scale").as_double();
  config_.imu_rotation_weight =
      this->get_parameter("visual_odometry_node.imu_rotation_weight")
          .as_double();
}

void VisualOdometry::imuOdometryHandler(
    const nav_msgs::msg::Odometry::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(mtx_);
  const double t = secs(msg->header.stamp);
  Transformd T(Eigen::Quaterniond(msg->pose.pose.orientation.w,
                                  msg->pose.pose.orientation.x,
                                  msg->pose.pose.orientation.y,
                                  msg->pose.pose.orientation.z),
               Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y,
                               msg->pose.pose.position.z));
  T.rot.normalize();
  imu_odom_buf_[t] = T;
  while (imu_odom_buf_.size() > 3000) {
    imu_odom_buf_.erase(imu_odom_buf_.begin());
  }
}

bool VisualOdometry::getImuOdomAtTime(double timestamp, Transformd &T_w_i) const {
  std::lock_guard<std::mutex> lock(mtx_);
  if (imu_odom_buf_.empty()) {
    return false;
  }

  auto it_upper = imu_odom_buf_.lower_bound(timestamp);
  if (it_upper == imu_odom_buf_.begin()) {
    if (std::abs(it_upper->first - timestamp) > config_.max_time_diff_s) {
      return false;
    }
    T_w_i = it_upper->second;
    return true;
  }
  if (it_upper == imu_odom_buf_.end()) {
    const auto &tail = *imu_odom_buf_.rbegin();
    if (std::abs(tail.first - timestamp) > config_.max_time_diff_s) {
      return false;
    }
    T_w_i = tail.second;
    return true;
  }

  auto it_prev = std::prev(it_upper);
  if (std::abs(it_prev->first - timestamp) <= std::abs(it_upper->first - timestamp)) {
    if (std::abs(it_prev->first - timestamp) > config_.max_time_diff_s) {
      return false;
    }
    T_w_i = it_prev->second;
  } else {
    if (std::abs(it_upper->first - timestamp) > config_.max_time_diff_s) {
      return false;
    }
    T_w_i = it_upper->second;
  }
  return true;
}

bool VisualOdometry::estimateRelativePose(const cv::Mat &prev_img,
                                          const cv::Mat &curr_img,
                                          Eigen::Quaterniond &q_prev_curr,
                                          Eigen::Vector3d &t_prev_curr_dir,
                                          int &inliers,
                                          int &n_matches) {
  std::vector<cv::KeyPoint> kp_prev, kp_curr;
  cv::Mat desc_prev, desc_curr;
  orb_->detectAndCompute(prev_img, cv::noArray(), kp_prev, desc_prev);
  orb_->detectAndCompute(curr_img, cv::noArray(), kp_curr, desc_curr);

  if (desc_prev.empty() || desc_curr.empty()) {
    return false;
  }

  std::vector<std::vector<cv::DMatch>> knn_matches;
  matcher_.knnMatch(desc_prev, desc_curr, knn_matches, 2);
  std::vector<cv::DMatch> good_matches;
  good_matches.reserve(knn_matches.size());
  for (const auto &pair : knn_matches) {
    if (pair.size() < 2) {
      continue;
    }
    if (pair[0].distance < 0.75f * pair[1].distance) {
      good_matches.push_back(pair[0]);
    }
  }

  n_matches = static_cast<int>(good_matches.size());
  if (n_matches < config_.min_matches) {
    return false;
  }

  std::vector<cv::Point2f> pts_prev, pts_curr;
  pts_prev.reserve(good_matches.size());
  pts_curr.reserve(good_matches.size());
  for (const auto &m : good_matches) {
    pts_prev.push_back(kp_prev[m.queryIdx].pt);
    pts_curr.push_back(kp_curr[m.trainIdx].pt);
  }

  cv::Mat K = (cv::Mat_<double>(3, 3) << config_.fx, 0.0, config_.cx, 0.0,
               config_.fy, config_.cy, 0.0, 0.0, 1.0);
  cv::Mat inlier_mask;
  cv::Mat E = cv::findEssentialMat(pts_prev, pts_curr, K, cv::RANSAC, 0.999,
                                   1.0, inlier_mask);
  if (E.empty()) {
    return false;
  }

  cv::Mat R, t;
  inliers = cv::recoverPose(E, pts_prev, pts_curr, K, R, t, inlier_mask);
  if (inliers < config_.min_inliers) {
    return false;
  }

  Eigen::Matrix3d R_eig;
  R_eig << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
      R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), R.at<double>(2, 0),
      R.at<double>(2, 1), R.at<double>(2, 2);
  q_prev_curr = Eigen::Quaterniond(R_eig).normalized();
  t_prev_curr_dir =
      Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0));
  if (t_prev_curr_dir.norm() < 1e-6) {
    return false;
  }
  t_prev_curr_dir.normalize();
  return true;
}

void VisualOdometry::imageHandler(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
  } catch (const cv_bridge::Exception &) {
    try {
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception &e) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                           "[VisualOdometry] cv_bridge conversion failed: %s",
                           e.what());
      return;
    }
  }

  cv::Mat gray;
  if (cv_ptr->image.channels() == 1) {
    gray = cv_ptr->image;
  } else {
    cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);
  }
  if (gray.empty()) {
    return;
  }

  const double curr_t = secs(msg->header.stamp);
  Transformd T_w_i_curr;
  if (!getImuOdomAtTime(curr_t, T_w_i_curr)) {
    return;
  }

  if (!has_prev_frame_) {
    prev_gray_ = gray.clone();
    prev_img_time_ = curr_t;
    T_w_c_ = T_w_i_curr;
    has_prev_frame_ = true;
    has_pose_ = true;
    publishOdometry(curr_t);
    return;
  }

  Transformd T_w_i_prev;
  if (!getImuOdomAtTime(prev_img_time_, T_w_i_prev)) {
    prev_gray_ = gray.clone();
    prev_img_time_ = curr_t;
    return;
  }

  const double dt = std::max(1e-3, curr_t - prev_img_time_);
  const Transformd T_i_prev_curr = T_w_i_prev.inverse() * T_w_i_curr;
  const double imu_scale_raw = T_i_prev_curr.pos.norm();
  const double imu_speed = imu_scale_raw / dt;

  Eigen::Quaterniond q_vis_prev_curr = Eigen::Quaterniond::Identity();
  Eigen::Vector3d t_vis_dir = Eigen::Vector3d::Zero();
  int inliers = 0;
  int n_matches = 0;
  bool visual_ok = estimateRelativePose(prev_gray_, gray, q_vis_prev_curr, t_vis_dir,
                                        inliers, n_matches);

  Eigen::Quaterniond q_prev_curr = T_i_prev_curr.rot;
  Eigen::Vector3d t_prev_curr = T_i_prev_curr.pos;

  if (visual_ok) {
    const double ratio =
        std::min(1.0, static_cast<double>(inliers) / std::max(1, n_matches));
    const double imu_w = std::clamp(config_.imu_rotation_weight, 0.0, 1.0);
    const double blend_w = std::clamp(imu_w + (1.0 - ratio) * 0.5, 0.0, 1.0);
    q_prev_curr = T_i_prev_curr.rot.slerp(1.0 - blend_w, q_vis_prev_curr);
    q_prev_curr.normalize();

    double scale = imu_scale_raw;
    scale = std::clamp(scale, config_.min_imu_scale, config_.max_imu_scale);
    if (imu_speed > config_.max_imu_speed_for_scale) {
      scale = config_.min_imu_scale;
    }
    t_prev_curr = t_vis_dir * scale;
    if (t_prev_curr.norm() / dt > config_.max_visual_speed) {
      visual_ok = false;
    }
  }

  if (!visual_ok && !has_pose_) {
    prev_gray_ = gray.clone();
    prev_img_time_ = curr_t;
    return;
  }

  if (has_pose_) {
    T_w_c_ = T_w_c_ * Transformd(q_prev_curr, t_prev_curr);
  } else {
    T_w_c_ = T_w_i_curr;
    has_pose_ = true;
  }

  publishOdometry(curr_t);
  prev_gray_ = gray.clone();
  prev_img_time_ = curr_t;
}

void VisualOdometry::publishOdometry(double timestamp) {
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = rclcpp::Time(timestamp * 1e9);
  odom.header.frame_id = config_.world_frame;
  odom.child_frame_id = config_.camera_frame;

  odom.pose.pose.position.x = T_w_c_.pos.x();
  odom.pose.pose.position.y = T_w_c_.pos.y();
  odom.pose.pose.position.z = T_w_c_.pos.z();
  odom.pose.pose.orientation.x = T_w_c_.rot.x();
  odom.pose.pose.orientation.y = T_w_c_.rot.y();
  odom.pose.pose.orientation.z = T_w_c_.rot.z();
  odom.pose.pose.orientation.w = T_w_c_.rot.w();

  pub_visual_odom_->publish(odom);
}

double VisualOdometry::secs(const builtin_interfaces::msg::Time &stamp) {
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

} // namespace super_odometry
