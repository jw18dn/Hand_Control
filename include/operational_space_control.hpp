#pragma once
#include <Eigen/Dense>
#include <mujoco/mujoco.h>
#include <vector>
#include "body_enum.hpp"
#include "states.hpp"
#include "parse_data.hpp"
#include "kinematics.hpp"


class OSCControl {
private:
    Kinematics& kinematics;
    States& states;
    ParseData& csvData;

    // Task weights for each body
    double w_palm_orient_   = 10000.0;  // Palm orientation weight (prioritize palm)
    double w_thumb_         = 5.0;      // Thumb position weight (reduced to prioritize palm)
    double w_index_         = 5.0;      // Index finger position weight (reduced to prioritize palm)
    double w_middle_        = 5.0;      // Middle finger position weight (reduced to prioritize palm)

    // Regularization weights
    double w_regularization_= 5.0;      // Acceleration regularization (smoothing)
    double w_torque_limit_  = 0.0;      // Soft torque limit weight

    // Other weights
    double w_coupling       = 1e0;      // Coupling
    double w_collision      = 0.0;      // Collision avoidance weight
    double w_palm_limit     = 100.0;    // Palm motion limiting weight

    // Control gains for desired accelerations
    double kp_palm_orient_  = 30.0;     // Palm orientation proportional gain
    double kd_palm_orient_  = 30.0;     // Palm orientation derivative gain
    double kp_finger_       = 10.0;
    double kd_finger_       = 3.0;

    // Limits
    double max_palm_velocity = 0.0;     // rad/s
    double max_palm_acceleration = 2.0; // rad/s^2

    // Other 
    double collision_margin = 0.0; // rad margin from limits
    double repulsion_gain = 0.0;

public:
    OSCControl(Kinematics& kin, States& st, ParseData& csv): kinematics(kin), states(st), csvData(csv) {}

    /**
     * Compute optimal control torques for the entire hand system
     */
    Eigen::VectorXd computeJointTorques(
        const mjModel* m, mjData* d,
        int csv_idx
    );

    /**
     * Compute reaction torques on palm from finger dynamics
     * @param m MuJoCo model
     * @param d MuJoCo data
     * @param tau_whole_hand Full torque vector from whole hand QP
     * @return 3D torque vector on palm in palm body frame
     */
    Eigen::Vector3d computePalmTorquesFromFingers(
        const mjModel* m,
        mjData* d,
        const Eigen::VectorXd& tau_whole_hand
    );

private:
    /**
     * Compute desired palm orientation acceleration (in body frame)
     */
    Eigen::Vector3d computePalmOrientAcceleration(
        const mjModel* m, mjData* d,
        const Eigen::Quaterniond& q_des,
        const Eigen::Quaterniond& q_cur,
        const Eigen::Vector3d& omega_cur
    );

    /**
     * Compute desired finger position acceleration
     */
    Eigen::Vector3d computeFingerPosAcceleration(
        const Eigen::Vector3d& pos_des,
        const Eigen::Vector3d& pos_cur,
        const Eigen::Vector3d& vel_cur
    );

    /**
     * Compute orientation error in body frame (for palm control)
     */
    Eigen::Vector3d computeOrientationError(
        const Eigen::Quaterniond& q_des,
        const Eigen::Quaterniond& q_cur
    );
};
