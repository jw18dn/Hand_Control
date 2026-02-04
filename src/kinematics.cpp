#include "kinematics.hpp"

Kinematics::Kinematics() {}

Eigen::Matrix3d Kinematics::vec2Skew(const Eigen::Vector3d &vec) {
    Eigen::Matrix3d skew;
    skew <<     0.0, -vec.z(),  vec.y(),
          vec.z(),     0.0, -vec.x(),
         -vec.y(),  vec.x(),     0.0;
    return skew;
}

// Compute Jacobian in world frame
Eigen::MatrixXd Kinematics::computeJacobian_W(const mjModel* m, const mjData* d, const Body& body_enum)
{
    int body_id = getBodyRef(m, body_enum);
    int nv = m->nv;

    // Resize buffers only if needed (optimization: reuse existing allocation)
    if (jacp_buffer_.size() != static_cast<size_t>(3 * nv)) {
        jacp_buffer_.resize(3 * nv);
        jacr_buffer_.resize(3 * nv);
    }

    // Zero out buffers
    std::fill(jacp_buffer_.begin(), jacp_buffer_.end(), 0.0);
    std::fill(jacr_buffer_.begin(), jacr_buffer_.end(), 0.0);

    // Compute Jacobian using MuJoCo
    mj_jacBody(m, d, jacp_buffer_.data(), jacr_buffer_.data(), body_id);

    // Allocate output matrix (6 rows: 3 for linear velocity, 3 for angular velocity)
    Eigen::MatrixXd J_world(6, nv);

    // ROW-MAJOR (default - MuJoCo fills jacp/jacr as 3×nv matrices in row-major order)
    Eigen::Map<Eigen::Matrix<mjtNum, 3, Eigen::Dynamic, Eigen::RowMajor>> jacp_map(jacp_buffer_.data(), 3, nv);
    Eigen::Map<Eigen::Matrix<mjtNum, 3, Eigen::Dynamic, Eigen::RowMajor>> jacr_map(jacr_buffer_.data(), 3, nv);

    // Copy to output matrix
    J_world.block(0, 0, 3, nv) = jacp_map.cast<double>();  // Linear velocity Jacobian
    J_world.block(3, 0, 3, nv) = jacr_map.cast<double>();  // Angular velocity Jacobian

    return J_world;
}

// Transform Jacobian from world frame to a reference frame
Eigen::MatrixXd Kinematics::transformJacobian_W_to_B(const mjModel* m, const mjData* d, const Eigen::MatrixXd& J_world, const Body& body_enum, const Body& ref_enum)
{
    int body_id = getBodyRef(m, body_enum);
    int ref_id = getBodyRef(m, ref_enum);
    int nq = J_world.cols();

    // Get rotation matrix of reference frame
    Eigen::Matrix3d R_ref;
    const mjtNum* R = d->xmat + 9*ref_id; // 3x3 rotation matrix column-major
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R_ref(r,c) = R[3*c + r]; // convert column-major to row-major Eigen

    // Body positions
    Eigen::Vector3d p_body(d->xpos[3*body_id+0], d->xpos[3*body_id+1], d->xpos[3*body_id+2]);
    Eigen::Vector3d p_ref(d->xpos[3*ref_id+0], d->xpos[3*ref_id+1], d->xpos[3*ref_id+2]);

    // Transform each column of the Jacobian
    Eigen::MatrixXd J_ref(6, nq);
    for (int col = 0; col < nq; ++col)
    {
        // Extract world-frame Jacobian column
        Eigen::Vector3d Jv_world = J_world.block<3,1>(0, col);
        Eigen::Vector3d Jw_world = J_world.block<3,1>(3, col);

        // Transform linear Jacobian to ref frame
        // Formula: Jv_ref = R_ref^T * (Jv_world - Jw_world × (p_body - p_ref))
        // The cross product accounts for the offset between body and reference frame
        Eigen::Vector3d Jv_ref = R_ref.transpose() * (Jv_world - Jw_world.cross(p_body - p_ref));

        // Transform angular Jacobian to ref frame
        // Formula: Jw_ref = R_ref^T * Jw_world
        Eigen::Vector3d Jw_ref = R_ref.transpose() * Jw_world;

        // Store in reference-frame Jacobian
        J_ref.block<3,1>(0, col) = Jv_ref;
        J_ref.block<3,1>(3, col) = Jw_ref;
    }

    return J_ref;
}

// Convenience wrapper: compute Jacobian directly in a reference frame
Eigen::MatrixXd Kinematics::computeJacobian(const mjModel* m, const mjData* d, const Body& body_enum, const Body& base_enum)
{
    // First get Jacobian in world frame
    Eigen::MatrixXd J_world = computeJacobian_W(m, d, body_enum);

    // If base is world/universe, return as-is
    if (base_enum == Body::UNIVERSE) {
        return J_world;
    }

    // Otherwise transform to the desired reference frame
    return transformJacobian_W_to_B(m, d, J_world, body_enum, base_enum);
}

double Kinematics::checkJacobianCondition(Eigen::MatrixXd J, double threshold){
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(J);
    double sigma_max = svd.singularValues()(0);
    double sigma_min = svd.singularValues()(svd.singularValues().size()-1);
    double cond_number = sigma_max / sigma_min;

    if (cond_number > threshold) {
        std::cout << "Warning: Jacobian is ill-conditioned! Condition number: " << cond_number << "\n";
    }

    return cond_number;
}

// Computes the Jacobian for a manipulator and returns a damped pseudoinverse with adaptive damping
Eigen::MatrixXd Kinematics::computeJacobianInv(const Eigen::MatrixXd& J, double base_damping, bool adaptive)
{
    int m = J.rows(); // task space DOF
    int n = J.cols(); // joint space DOF

    double damping = base_damping;

    // If adaptive damping is enabled, compute condition number and adjust damping
    if (adaptive) {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
        double sigma_max = svd.singularValues()(0);
        double sigma_min = svd.singularValues()(std::min(m, n) - 1);

        // Avoid division by zero
        if (sigma_min < 1e-10) {
            sigma_min = 1e-10;
        }

        double cond_number = sigma_max / sigma_min;

        // Adaptive damping: increase damping for ill-conditioned Jacobians
        // Formula: lambda = base_damping * sqrt(cond_number / threshold)
        constexpr double cond_threshold = 1e3;  // Start increasing damping at this condition number
        if (cond_number > cond_threshold) {
            damping = base_damping * std::sqrt(cond_number / cond_threshold);
            // Cap maximum damping to prevent over-damping
            constexpr double max_damping = 1.0;
            damping = std::min(damping, max_damping);
        }
    }

    // Compute damped least squares pseudoinverse:
    // J^+ = J^T * (J*J^T + lambda^2 * I)^-1
    Eigen::MatrixXd JJt = J * J.transpose();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(m, m);
    Eigen::MatrixXd damped_inv = (JJt + damping*damping * I).inverse();

    Eigen::MatrixXd J_pseudo = J.transpose() * damped_inv;
    return J_pseudo;
}


// Transform a pose from world frame to a reference frame
Eigen::Isometry3d Kinematics::transformIsometry_W_to_B(const Eigen::Isometry3d& T_body_world, const Eigen::Isometry3d& T_ref_world)
{
    // Compute: T_body_ref = T_ref_world^(-1) * T_body_world
    return T_ref_world.inverse() * T_body_world;
}

