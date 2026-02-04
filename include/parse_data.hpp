#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "mujoco_driver.hpp"
#include "body_enum.hpp"


class ParseData {
private:
    // Member Vars
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> data_;
    Eigen::Matrix<int64_t, Eigen::Dynamic, 1>  timestamps_;
    std::vector<Eigen::Isometry3d> palm_W_, thumb_W_, middle_W_, index_W_;
    std::vector<Eigen::Quaterniond> palm_W_quat_, thumb_W_quat_, middle_W_quat_, index_W_quat_;

    // Velocities (in body frame)
    std::vector<Eigen::Vector3d> palm_vel_B_, thumb_vel_B_, middle_vel_B_, index_vel_B_;
    std::vector<Eigen::Vector3d> palm_omega_B_, thumb_omega_B_, middle_omega_B_, index_omega_B_;

    // Accelerations (in body frame)
    std::vector<Eigen::Vector3d> palm_acc_B_, thumb_acc_B_, middle_acc_B_, index_acc_B_;
    std::vector<Eigen::Vector3d> palm_alpha_B_, thumb_alpha_B_, middle_alpha_B_, index_alpha_B_;

    // Helper functions
    void computeVelocitiesAndAccelerations();
    Eigen::Vector3d computeLinearVelocity(int idx, const std::vector<Eigen::Isometry3d>& poses, const std::vector<Eigen::Quaterniond>& quats) const;
    Eigen::Vector3d computeAngularVelocity(int idx, const std::vector<Eigen::Quaterniond>& quats) const;
    Eigen::Vector3d computeLinearAcceleration(int idx, const std::vector<Eigen::Vector3d>& velocities) const;
    Eigen::Vector3d computeAngularAcceleration(int idx, const std::vector<Eigen::Vector3d>& angular_vels) const;

public:
    // Constructor
    ParseData();

    // Time and indecies (change some of these to vectors)
    double getData_dt(int idx) const;
    int getRefIdx(const int64_t time_start, const int64_t curr_start, const Eigen::Matrix<int64_t, Eigen::Dynamic, 1>& timestamps) const;
    const Eigen::Matrix<int64_t, Eigen::Dynamic, 1>& getTimestamps() const noexcept {return timestamps_;}

    // Position (World Frame)
    const Eigen::Isometry3d& getBodyIso_des_W(int idx, Body body) const;
    const Eigen::Quaterniond& getBodyQuat_des_W(int idx, Body body) const;
    Eigen::Quaterniond getBodyQuat_des_W_at(const int64_t start_time, const int64_t curr_time, Body body) const;

    // Velocity (body frame)
    const Eigen::Vector3d& getBodyLinearVel_des_B(int idx, Body body) const;
    const Eigen::Vector3d& getBodyAngularVel_des_B(int idx, Body body) const;
    Eigen::Vector3d getBodyAngularVel_des_B_at(const int64_t start_time, const int64_t curr_time, Body body) const;

    // Acceleration (body frame)
    const Eigen::Vector3d& getBodyLinearAcc_des_B(int idx, Body body) const;
    const Eigen::Vector3d& getBodyAngularAcc_des_B(int idx, Body body) const;

    // Setters
    void loadCSV(const std::string& filename);
};