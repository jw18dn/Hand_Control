#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "body_enum.hpp"
#include <mujoco/mujoco.h>

class Kinematics {
private:
    // Reusable buffers for Jacobian computation (optimization to avoid allocations in hot path)
    mutable std::vector<mjtNum> jacp_buffer_;
    mutable std::vector<mjtNum> jacr_buffer_;

public:
    // Constructor
    Kinematics();

    // Skew symetric Matrix Utilities
    Eigen::Matrix3d vec2Skew(const Eigen::Vector3d &vec);
    Eigen::Vector3d skew2Vec(const Eigen::Matrix3d &skew) {return Eigen::Vector3d(skew(2,1), skew(0,2), skew(1,0));}

    // Jacobians
    Eigen::MatrixXd computeJacobian(const mjModel* m, const mjData* d, const Body& body_enum, const Body& base_enum);
    Eigen::MatrixXd computeJacobian_W(const mjModel* m, const mjData* d, const Body& body_enum);
    Eigen::MatrixXd transformJacobian_W_to_B(const mjModel* m, const mjData* d, const Eigen::MatrixXd& J_world, const Body& body_enum, const Body& ref_enum);
    double checkJacobianCondition(Eigen::MatrixXd J, double threshold = 1e4);

    // Computes the Jacobian for a manipulator and returns a damped pseudoinverse with adaptive damping
    Eigen::MatrixXd computeJacobianInv(const Eigen::MatrixXd& J, double base_damping = 0.01, bool adaptive = true);

    // Transform a pose from world frame to a reference frame
    Eigen::Isometry3d transformIsometry_W_to_B(const Eigen::Isometry3d& T_body_world, const Eigen::Isometry3d& T_ref_world);

};