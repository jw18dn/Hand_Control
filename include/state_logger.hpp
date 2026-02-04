#pragma once
#include <fstream>
#include <string>
#include <chrono>
#include <Eigen/Dense>
#include <mujoco/mujoco.h>
#include "body_enum.hpp"
#include "states.hpp"
#include "parse_data.hpp"

class StateLogger {
private:
    std::ofstream log_file_;
    bool initialized_ = false;
    int64_t start_time_;
    int log_counter_ = 0;
    int log_frequency_ = 10;  // Log every N control steps to avoid massive files

public:
    StateLogger() = default;

    // Initialize the logger with a filename
    bool initialize(const std::string& filename, int log_frequency = 10);

    // Log actual vs desired states for a specific body
    void logBodyState(
        const mjModel* m,
        const States& states,
        const ParseData& csvData,
        int csv_idx,
        Body body,
        const std::string& body_name
    );

    // Log position error
    void logPositionError(
        const Eigen::Vector3d& desired_pos,
        const Eigen::Vector3d& actual_pos,
        const std::string& body_name
    );

    // Log orientation error
    void logOrientationError(
        const Eigen::Quaterniond& desired_quat,
        const Eigen::Quaterniond& actual_quat,
        const std::string& body_name
    );

    // Log complete state (positions, velocities, errors)
    void logCompleteState(
        const mjModel* m,
        const mjData* d,
        const States& states,
        const ParseData& csvData,
        int csv_idx
    );

    // Manually flush the log
    void flush();

    // Close the log file
    void close();

    ~StateLogger() {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }
};
