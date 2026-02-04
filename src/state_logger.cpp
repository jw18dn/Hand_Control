#include "state_logger.hpp"
#include <iostream>
#include <iomanip>

bool StateLogger::initialize(const std::string& filename, int log_frequency) {
    if (initialized_) {
        std::cerr << "StateLogger already initialized\n";
        return false;
    }

    log_frequency_ = log_frequency;
    log_file_.open(filename);

    if (!log_file_.is_open()) {
        std::cerr << "Failed to open log file: " << filename << "\n";
        return false;
    }

    // Write CSV header
    log_file_ << "timestamp_ms,csv_idx,body_name,"
              << "des_pos_x,des_pos_y,des_pos_z,"
              << "act_pos_x,act_pos_y,act_pos_z,"
              << "pos_err_x,pos_err_y,pos_err_z,pos_err_norm,"
              << "des_quat_w,des_quat_x,des_quat_y,des_quat_z,"
              << "act_quat_w,act_quat_x,act_quat_y,act_quat_z,"
              << "orient_err_angle\n";

    start_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    initialized_ = true;
    return true;
}

void StateLogger::logBodyState(
    const mjModel* m,
    const States& states,
    const ParseData& csvData,
    int csv_idx,
    Body body,
    const std::string& body_name
) {
    if (!initialized_) return;

    // Get desired and actual states (with error handling)
    Eigen::Isometry3d desired_iso;
    Eigen::Quaterniond desired_quat;
    Eigen::Isometry3d actual_iso;
    Eigen::Quaterniond actual_quat;

    try {
        desired_iso = csvData.getBodyIso_des_W(csv_idx, body);
        desired_quat = csvData.getBodyQuat_des_W(csv_idx, body);
        actual_iso = states.getBodyIso_meas_W(m, body);
        actual_quat = states.getBodyQuat_meas_W(m, body);
    } catch (const std::exception& e) {
        std::cerr << "Error getting state for " << body_name << ": " << e.what() << "\n";
        return;
    }

    // Compute errors
    Eigen::Vector3d pos_error = desired_iso.translation() - actual_iso.translation();
    double pos_err_norm = pos_error.norm();

    // Orientation error (angle between quaternions)
    Eigen::Quaterniond quat_error = desired_quat * actual_quat.inverse();
    double orient_err_angle = 2.0 * std::acos(std::min(1.0, std::abs(quat_error.w())));

    // Get current timestamp
    int64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    int64_t elapsed_ms = current_time - start_time_;

    // Write to CSV
    log_file_ << std::fixed << std::setprecision(6)
              << elapsed_ms << ","
              << csv_idx << ","
              << body_name << ","
              << desired_iso.translation().x() << ","
              << desired_iso.translation().y() << ","
              << desired_iso.translation().z() << ","
              << actual_iso.translation().x() << ","
              << actual_iso.translation().y() << ","
              << actual_iso.translation().z() << ","
              << pos_error.x() << ","
              << pos_error.y() << ","
              << pos_error.z() << ","
              << pos_err_norm << ","
              << desired_quat.w() << ","
              << desired_quat.x() << ","
              << desired_quat.y() << ","
              << desired_quat.z() << ","
              << actual_quat.w() << ","
              << actual_quat.x() << ","
              << actual_quat.y() << ","
              << actual_quat.z() << ","
              << orient_err_angle << "\n";
}

void StateLogger::logPositionError(
    const Eigen::Vector3d& desired_pos,
    const Eigen::Vector3d& actual_pos,
    const std::string& body_name
) {
    if (!initialized_) return;

    Eigen::Vector3d error = desired_pos - actual_pos;

    int64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    int64_t elapsed_ms = current_time - start_time_;

    std::cout << "[" << elapsed_ms << "ms] " << body_name
              << " Position Error: [" << error.transpose()
              << "] norm=" << error.norm() << "\n";
}

void StateLogger::logOrientationError(
    const Eigen::Quaterniond& desired_quat,
    const Eigen::Quaterniond& actual_quat,
    const std::string& body_name
) {
    if (!initialized_) return;

    Eigen::Quaterniond quat_error = desired_quat * actual_quat.inverse();
    double angle_error = 2.0 * std::acos(std::min(1.0, std::abs(quat_error.w())));

    int64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    int64_t elapsed_ms = current_time - start_time_;

    std::cout << "[" << elapsed_ms << "ms] " << body_name
              << " Orientation Error: " << angle_error << " rad ("
              << (angle_error * 180.0 / M_PI) << " deg)\n";
}

void StateLogger::logCompleteState(
    const mjModel* m,
    const mjData* d,
    const States& states,
    const ParseData& csvData,
    int csv_idx
) {
    if (!initialized_) return;

    // Only log every N steps to reduce file size (check once for all bodies)
    log_counter_++;
    if (log_counter_ % log_frequency_ != 0) return;

    // Log all tracked bodies
    logBodyState(m, states, csvData, csv_idx, Body::PALM, "PALM");
    logBodyState(m, states, csvData, csv_idx, Body::THUMB_EE, "THUMB_EE");
    logBodyState(m, states, csvData, csv_idx, Body::INDEX_EE, "INDEX_EE");
    logBodyState(m, states, csvData, csv_idx, Body::MIDDLE_EE, "MIDDLE_EE");
}

void StateLogger::flush() {
    if (log_file_.is_open()) {
        log_file_.flush();
    }
}

void StateLogger::close() {
    if (log_file_.is_open()) {
        log_file_.close();
        initialized_ = false;
    }
}
