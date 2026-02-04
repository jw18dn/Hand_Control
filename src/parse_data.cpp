#include "parse_data.hpp"
using namespace std::chrono;

ParseData::ParseData() {}

void ParseData::loadCSV(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    std::getline(infile, line); // skip header

    // Temporary storage to count rows and columns
    std::vector<long long> temp_timestamps;
    std::vector<std::vector<double>> temp_data;

    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string item;

        // Timestamp
        std::getline(ss, item, ',');
        temp_timestamps.push_back(std::stoll(item));

        // Skip index column
        std::getline(ss, item, ',');

        // Remaining columns (X_1, Y_1, Z_1, qX_1, qY_1, qZ_1, qW_1, X_1, Y_1, Z_1, qX_1, qY_1, qZ_1, qW_1)
        std::vector<double> row_vals;
        while (std::getline(ss, item, ',')) {
            row_vals.push_back(std::stod(item));
        }

        temp_data.push_back(row_vals);
    }

    infile.close();

    // Convert timestamps to Eigen::VectorXi
    size_t n_rows = temp_data.size();
    if (n_rows == 0) {
        throw std::runtime_error("CSV file is empty or contains no data rows: " + filename);
    }

    size_t n_cols = temp_data[0].size();

    // Resize the timestamp vec & data mat
    data_.resize(n_rows, n_cols);
    timestamps_.resize(n_rows);

    // Resize the isometry matricies
    palm_W_.resize(n_rows);  
    thumb_W_.resize(n_rows);  
    middle_W_.resize(n_rows);  
    index_W_.resize(n_rows);  
    palm_W_quat_.resize(n_rows);  
    thumb_W_quat_.resize(n_rows);  
    middle_W_quat_.resize(n_rows);  
    index_W_quat_.resize(n_rows);  

    // Assign Values
    for (size_t i = 0; i < n_rows; ++i) {
        // Timestep
        timestamps_(i) = static_cast<int64_t>(temp_timestamps[i]);

        // All data
        for (size_t j = 0; j < n_cols; ++j) {
            data_(i, j) = temp_data[i][j];
        }

        // Body Translational data (CSV order: Palm, Thumb, Index, Middle)
        palm_W_[i].translation() = data_.row(i).segment<3>(0);
        thumb_W_[i].translation() = data_.row(i).segment<3>(7);
        index_W_[i].translation() = data_.row(i).segment<3>(14);   // Fixed: was 21
        middle_W_[i].translation() = data_.row(i).segment<3>(21);  // Fixed: was 14

        // Rotational Data (Quaternion format: w, x, y, z)
        palm_W_quat_[i] = Eigen::Quaterniond(data_(i,6), data_(i,3), data_(i,4), data_(i,5));
        // palm_W_quat_[i] = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        palm_W_[i].linear() = palm_W_quat_[i].toRotationMatrix();

        thumb_W_quat_[i] = Eigen::Quaterniond(data_(i,13), data_(i,10), data_(i,11), data_(i,12));
        thumb_W_[i].linear() = thumb_W_quat_[i].toRotationMatrix();

        index_W_quat_[i] = Eigen::Quaterniond(data_(i,20), data_(i,17), data_(i,18), data_(i,19));  // Fixed: was 27,24,25,26
        index_W_[i].linear() = index_W_quat_[i].toRotationMatrix();

        middle_W_quat_[i] = Eigen::Quaterniond(data_(i,27), data_(i,24), data_(i,25), data_(i,26));  // Fixed: was 20,17,18,19
        middle_W_[i].linear() = middle_W_quat_[i].toRotationMatrix();
    }

    // Compute velocities and accelerations
    computeVelocitiesAndAccelerations();
}

int ParseData::getRefIdx(const int64_t time_start, const int64_t curr_start, const Eigen::Matrix<int64_t, Eigen::Dynamic, 1>& timestamps) const
{
    // Use simulation time instead of wall clock time
    int64_t time_diff = curr_start - time_start;
    int64_t csv_time = timestamps[0] + time_diff;

    int ref_pos = 0;

    // Check bounds
    if (csv_time <= timestamps[0]) {
        ref_pos = 0;
    } else if (csv_time >= timestamps[timestamps.size() - 1]) {
        ref_pos = timestamps.size() - 1;
    } else {
        // Find first index where timestamp >= csv_time
        for (int64_t i = 0; i < timestamps.size(); ++i) {
            if (timestamps[i] >= csv_time) {
                ref_pos = i;
                break;
            }
        }
    }

    return ref_pos;
}

double ParseData::getData_dt(int idx) const
{
    // Bounds checking
    if (idx < 0 || idx >= static_cast<int>(timestamps_.size())) {
        throw std::out_of_range("getDt: idx out of range");
    }

    if (timestamps_.size() < 3) {
        throw std::runtime_error("getDt: insufficient data (need at least 3 timestamps)");
    }

    if (idx == 0)
    {
        double time_milli = timestamps_[idx+2] - timestamps_[idx];
        return time_milli/2000;
    }
    else if (idx >= timestamps_.size() - 1)
    {
        double time_milli = timestamps_[timestamps_.size()-1] - timestamps_[timestamps_.size()-3];
        return time_milli/2000;
    }
    else
    {
        double time_milli = timestamps_[idx+1] - timestamps_[idx-1];
        return time_milli/2000;
    }
}


// Map Body enum to the Isometry vector
const Eigen::Isometry3d& ParseData::getBodyIso_des_W(int idx, Body body) const {
    // Bounds checking
    if (idx < 0 || idx >= static_cast<int>(palm_W_.size())) {
        throw std::out_of_range("getBodyIso_des_W: idx out of range");
    }

    switch (body) {
        case Body::PALM:        return palm_W_[idx];
        case Body::THUMB_EE:    return thumb_W_[idx];
        case Body::MIDDLE_EE:   return middle_W_[idx];
        case Body::INDEX_EE:    return index_W_[idx];
        default:                throw std::runtime_error("Unknown Body enum");
    }
}

// Map Body enum to the Quaternion vector
const Eigen::Quaterniond& ParseData::getBodyQuat_des_W(int idx, Body body) const {
    // Bounds checking
    if (idx < 0 || idx >= static_cast<int>(palm_W_quat_.size())) {
        throw std::out_of_range("getBodyQuat_des_W: idx out of range");
    }

    switch (body) {
        case Body::PALM:        return palm_W_quat_[idx];
        case Body::THUMB_EE:    return thumb_W_quat_[idx];
        case Body::MIDDLE_EE:   return middle_W_quat_[idx];
        case Body::INDEX_EE:    return index_W_quat_[idx];
        default:                throw std::runtime_error("Unknown Body enum");
    }
}

Eigen::Quaterniond ParseData::getBodyQuat_des_W_at(int64_t start_time, int64_t curr_time, Body body) const {

    // Find the index in timestamps
    int idx = getRefIdx(start_time, curr_time, timestamps_);

    // Check if we're at or beyond the last index - if so, return the last value
    if (idx >= timestamps_.size() - 1) {
        return getBodyQuat_des_W(timestamps_.size() - 1, body);
    }

    int64_t time1 = timestamps_[idx];
    int64_t time2 = timestamps_[idx+1];

    // Get the quaternions at the two bounding times
    Eigen::Quaterniond quat1 = getBodyQuat_des_W(idx, body);   // at time1
    Eigen::Quaterniond quat2 = getBodyQuat_des_W(idx+1, body); // at time2

    // Compute interpolation factor (clamped between 0 and 1)
    double alpha = 0.0;
    if (time2 != time1) {  // avoid division by zero
        alpha = double(curr_time - time1) / double(time2 - time1);
        alpha = std::clamp(alpha, 0.0, 1.0);
    }

    // Spherical linear interpolation
    Eigen::Quaterniond quat_interp = quat1.slerp(alpha, quat2);

    return quat_interp;
}


// Get desired linear velocity (body frame)
const Eigen::Vector3d& ParseData::getBodyLinearVel_des_B(int idx, Body body) const {
    if (idx < 0 || idx >= static_cast<int>(palm_vel_B_.size())) {
        throw std::out_of_range("getBodyLinearVel_des_B: idx out of range");
    }

    switch (body) {
        case Body::PALM:        return palm_vel_B_[idx];
        case Body::THUMB_EE:    return thumb_vel_B_[idx];
        case Body::MIDDLE_EE:   return middle_vel_B_[idx];
        case Body::INDEX_EE:    return index_vel_B_[idx];
        default:                throw std::runtime_error("Unknown Body enum");
    }
}

// Get desired angular velocity (body frame)
const Eigen::Vector3d& ParseData::getBodyAngularVel_des_B(int idx, Body body) const {
    if (idx < 0 || idx >= static_cast<int>(palm_omega_B_.size())) {
        throw std::out_of_range("getBodyAngularVel_des_B: idx out of range");
    }

    switch (body) {
        case Body::PALM:        return palm_omega_B_[idx];
        case Body::THUMB_EE:    return thumb_omega_B_[idx];
        case Body::MIDDLE_EE:   return middle_omega_B_[idx];
        case Body::INDEX_EE:    return index_omega_B_[idx];
        default:                throw std::runtime_error("Unknown Body enum");
    }
}
Eigen::Vector3d ParseData::getBodyAngularVel_des_B_at(int64_t start_time, int64_t curr_time, Body body) const {

    // Find the index in timestamps
    int idx = getRefIdx(start_time, curr_time, timestamps_);

    // Check if we're at or beyond the last index - if so, return the last value
    if (idx >= timestamps_.size() - 1) {
        return getBodyAngularVel_des_B(timestamps_.size() - 1, body);
    }

    int64_t time1 = timestamps_[idx];
    int64_t time2 = timestamps_[idx+1];

    // Get the quaternions at the two bounding times
    Eigen::Vector3d omega1 = getBodyAngularVel_des_B(idx, body);   // at time1
    Eigen::Vector3d omega2 = getBodyAngularVel_des_B(idx+1, body); // at time2

    // Compute interpolation factor (clamped between 0 and 1)
    double alpha = 0.0;
    if (time2 != time1) {  // avoid division by zero
        alpha = double(curr_time - time1) / double(time2 - time1);
        alpha = std::clamp(alpha, 0.0, 1.0);
    }

    // Spherical linear interpolation
    Eigen::Vector3d omega_interp = (1.0 - alpha) * omega1 + alpha * omega2;

    return omega_interp;
}


// Get desired linear acceleration (body frame)
const Eigen::Vector3d& ParseData::getBodyLinearAcc_des_B(int idx, Body body) const {
    if (idx < 0 || idx >= static_cast<int>(palm_acc_B_.size())) {
        throw std::out_of_range("getBodyLinearAcc_des_B: idx out of range");
    }

    switch (body) {
        case Body::PALM:        return palm_acc_B_[idx];
        case Body::THUMB_EE:    return thumb_acc_B_[idx];
        case Body::MIDDLE_EE:   return middle_acc_B_[idx];
        case Body::INDEX_EE:    return index_acc_B_[idx];
        default:                throw std::runtime_error("Unknown Body enum");
    }
}

// Get desired angular acceleration (body frame)
const Eigen::Vector3d& ParseData::getBodyAngularAcc_des_B(int idx, Body body) const {
    if (idx < 0 || idx >= static_cast<int>(palm_alpha_B_.size())) {
        throw std::out_of_range("getBodyAngularAcc_des_B: idx out of range");
    }

    switch (body) {
        case Body::PALM:        return palm_alpha_B_[idx];
        case Body::THUMB_EE:    return thumb_alpha_B_[idx];
        case Body::MIDDLE_EE:   return middle_alpha_B_[idx];
        case Body::INDEX_EE:    return index_alpha_B_[idx];
        default:                throw std::runtime_error("Unknown Body enum");
    }
}

// Compute all velocities and accelerations using finite differences
void ParseData::computeVelocitiesAndAccelerations() {
    size_t n = palm_W_.size();

    // Resize all velocity and acceleration vectors
    palm_vel_B_.resize(n);
    thumb_vel_B_.resize(n);
    middle_vel_B_.resize(n);
    index_vel_B_.resize(n);

    palm_omega_B_.resize(n);
    thumb_omega_B_.resize(n);
    middle_omega_B_.resize(n);
    index_omega_B_.resize(n);

    palm_acc_B_.resize(n);
    thumb_acc_B_.resize(n);
    middle_acc_B_.resize(n);
    index_acc_B_.resize(n);

    palm_alpha_B_.resize(n);
    thumb_alpha_B_.resize(n);
    middle_alpha_B_.resize(n);
    index_alpha_B_.resize(n);

    // Compute velocities for all bodies
    for (size_t i = 0; i < n; ++i) {
        palm_vel_B_[i] = computeLinearVelocity(i, palm_W_, palm_W_quat_);
        thumb_vel_B_[i] = computeLinearVelocity(i, thumb_W_, thumb_W_quat_);
        middle_vel_B_[i] = computeLinearVelocity(i, middle_W_, middle_W_quat_);
        index_vel_B_[i] = computeLinearVelocity(i, index_W_, index_W_quat_);

        palm_omega_B_[i] = computeAngularVelocity(i, palm_W_quat_);
        thumb_omega_B_[i] = computeAngularVelocity(i, thumb_W_quat_);
        middle_omega_B_[i] = computeAngularVelocity(i, middle_W_quat_);
        index_omega_B_[i] = computeAngularVelocity(i, index_W_quat_);
    }

    // Compute accelerations from velocities
    for (size_t i = 0; i < n; ++i) {
        palm_acc_B_[i] = computeLinearAcceleration(i, palm_vel_B_);
        thumb_acc_B_[i] = computeLinearAcceleration(i, thumb_vel_B_);
        middle_acc_B_[i] = computeLinearAcceleration(i, middle_vel_B_);
        index_acc_B_[i] = computeLinearAcceleration(i, index_vel_B_);

        palm_alpha_B_[i] = computeAngularAcceleration(i, palm_omega_B_);
        thumb_alpha_B_[i] = computeAngularAcceleration(i, thumb_omega_B_);
        middle_alpha_B_[i] = computeAngularAcceleration(i, middle_omega_B_);
        index_alpha_B_[i] = computeAngularAcceleration(i, index_omega_B_);
    }
}

// Compute linear velocity in body frame using central differences
Eigen::Vector3d ParseData::computeLinearVelocity(int idx, const std::vector<Eigen::Isometry3d>& poses, const std::vector<Eigen::Quaterniond>& quats) const {
    size_t n = poses.size();

    Eigen::Vector3d vel_world;

    if (idx == 0) {
        // Forward difference
        double dt = (timestamps_[1] - timestamps_[0]) / 1000.0;  // Convert ms to seconds
        vel_world = (poses[1].translation() - poses[0].translation()) / dt;
    } else if (idx == static_cast<int>(n) - 1) {
        // Backward difference
        double dt = (timestamps_[n-1] - timestamps_[n-2]) / 1000.0;
        vel_world = (poses[n-1].translation() - poses[n-2].translation()) / dt;
    } else {
        // Central difference
        double dt = (timestamps_[idx+1] - timestamps_[idx-1]) / 1000.0;
        vel_world = (poses[idx+1].translation() - poses[idx-1].translation()) / dt;
    }

    // Transform to body frame
    Eigen::Matrix3d R_body = quats[idx].toRotationMatrix();
    return R_body.transpose() * vel_world;
}

// Compute angular velocity in body frame using quaternion derivatives
Eigen::Vector3d ParseData::computeAngularVelocity(int idx, const std::vector<Eigen::Quaterniond>& quats) const {
    size_t n = quats.size();

    Eigen::Quaterniond q_cur = quats[idx];
    Eigen::Quaterniond q_dot;

    if (idx == 0) {
        // Forward difference
        double dt = (timestamps_[1] - timestamps_[0]) / 1000.0;
        Eigen::Quaterniond q_next = quats[1];
        // Ensure shortest path
        if (q_cur.dot(q_next) < 0) q_next.coeffs() = -q_next.coeffs();
        q_dot.coeffs() = (q_next.coeffs() - q_cur.coeffs()) / dt;
    } else if (idx == static_cast<int>(n) - 1) {
        // Backward difference
        double dt = (timestamps_[n-1] - timestamps_[n-2]) / 1000.0;
        Eigen::Quaterniond q_prev = quats[n-2];
        if (q_cur.dot(q_prev) < 0) q_prev.coeffs() = -q_prev.coeffs();
        q_dot.coeffs() = (q_cur.coeffs() - q_prev.coeffs()) / dt;
    } else {
        // Central difference
        double dt = (timestamps_[idx+1] - timestamps_[idx-1]) / 1000.0;
        Eigen::Quaterniond q_next = quats[idx+1];
        Eigen::Quaterniond q_prev = quats[idx-1];
        if (q_cur.dot(q_next) < 0) q_next.coeffs() = -q_next.coeffs();
        if (q_cur.dot(q_prev) < 0) q_prev.coeffs() = -q_prev.coeffs();
        q_dot.coeffs() = (q_next.coeffs() - q_prev.coeffs()) / dt;
    }

    // Angular velocity in body frame: omega = 2 * q_dot * q_cur^{-1}
    // This gives a quaternion [0, omega_x, omega_y, omega_z] (pure imaginary)
    Eigen::Quaterniond omega_quat = q_dot * q_cur.inverse();

    // Extract the vector part and multiply by 2
    return 2.0 * Eigen::Vector3d(omega_quat.x(), omega_quat.y(), omega_quat.z());
}

// Compute linear acceleration in body frame from velocities
Eigen::Vector3d ParseData::computeLinearAcceleration(int idx, const std::vector<Eigen::Vector3d>& velocities) const {
    size_t n = velocities.size();

    if (idx == 0) {
        // Forward difference
        double dt = (timestamps_[1] - timestamps_[0]) / 1000.0;
        return (velocities[1] - velocities[0]) / dt;
    } else if (idx == static_cast<int>(n) - 1) {
        // Backward difference
        double dt = (timestamps_[n-1] - timestamps_[n-2]) / 1000.0;
        return (velocities[n-1] - velocities[n-2]) / dt;
    } else {
        // Central difference
        double dt = (timestamps_[idx+1] - timestamps_[idx-1]) / 1000.0;
        return (velocities[idx+1] - velocities[idx-1]) / dt;
    }
}

// Compute angular acceleration in body frame from angular velocities
Eigen::Vector3d ParseData::computeAngularAcceleration(int idx, const std::vector<Eigen::Vector3d>& angular_vels) const {
    size_t n = angular_vels.size();

    if (idx == 0) {
        // Forward difference
        double dt = (timestamps_[1] - timestamps_[0]) / 1000.0;
        return (angular_vels[1] - angular_vels[0]) / dt;
    } else if (idx == static_cast<int>(n) - 1) {
        // Backward difference
        double dt = (timestamps_[n-1] - timestamps_[n-2]) / 1000.0;
        return (angular_vels[n-1] - angular_vels[n-2]) / dt;
    } else {
        // Central difference
        double dt = (timestamps_[idx+1] - timestamps_[idx-1]) / 1000.0;
        return (angular_vels[idx+1] - angular_vels[idx-1]) / dt;
    }
}
