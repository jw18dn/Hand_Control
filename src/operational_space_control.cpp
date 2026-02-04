#include "operational_space_control.hpp"
#include "mujoco_api.hpp"
#include <iostream>
#include <Eigen/Sparse>

static int const PALM_DOF = 3;

Eigen::VectorXd OSCControl::computeJointTorques(const mjModel* m, mjData* d,int csv_idx)
{
    int nv = m->nv;

    // Get desired and current states for all bodies
    // Palm
    Eigen::Isometry3d palm_iso_des = csvData.getBodyIso_des_W(csv_idx, Body::PALM);
    Eigen::Isometry3d palm_iso_cur = states.getBodyIso_meas_W(m, Body::PALM);
    Eigen::Vector3d palm_pos_des = palm_iso_des.translation();
    Eigen::Vector3d palm_pos_cur = palm_iso_cur.translation();
    Eigen::Quaterniond palm_q_des = csvData.getBodyQuat_des_W(csv_idx, Body::PALM);
    Eigen::Quaterniond palm_q_cur = states.getBodyQuat_meas_W(m, Body::PALM);

    // Thumb
    Eigen::Isometry3d thumb_des_W = csvData.getBodyIso_des_W(csv_idx, Body::THUMB_EE);
    Eigen::Isometry3d thumb_cur_W = states.getBodyIso_meas_W(m, Body::THUMB_EE);
    Eigen::Isometry3d palm_iso = states.getBodyIso_meas_W(m, Body::PALM);
    Eigen::Isometry3d thumb_des = kinematics.transformIsometry_W_to_B(thumb_des_W, palm_iso);
    Eigen::Isometry3d thumb_cur = kinematics.transformIsometry_W_to_B(thumb_cur_W, palm_iso);

    // Index
    Eigen::Isometry3d index_des_W = csvData.getBodyIso_des_W(csv_idx, Body::INDEX_EE);
    Eigen::Isometry3d index_cur_W = states.getBodyIso_meas_W(m, Body::INDEX_EE);
    Eigen::Isometry3d index_des = kinematics.transformIsometry_W_to_B(index_des_W, palm_iso);
    Eigen::Isometry3d index_cur = kinematics.transformIsometry_W_to_B(index_cur_W, palm_iso);

    // Middle
    Eigen::Isometry3d middle_des_W = csvData.getBodyIso_des_W(csv_idx, Body::MIDDLE_EE);
    Eigen::Isometry3d middle_cur_W = states.getBodyIso_meas_W(m, Body::MIDDLE_EE);
    Eigen::Isometry3d middle_des = kinematics.transformIsometry_W_to_B(middle_des_W, palm_iso);
    Eigen::Isometry3d middle_cur = kinematics.transformIsometry_W_to_B(middle_cur_W, palm_iso);

    // Get joint velocities
    Eigen::VectorXd dq(nv);
    for (int i = 0; i < nv; i++) {
        dq[i] = d->qvel[i];
    }

    // Compute Jacobians
    // Palm position Jacobian (world frame, 3xnv)
    int palm_body_id = getBodyRef(m, Body::PALM);
    Eigen::MatrixXd J_palm_pos_W(3, nv);
    Eigen::MatrixXd J_palm_rot_W(3, nv);
    mj_jacBody(m, d, J_palm_pos_W.data(), J_palm_rot_W.data(), palm_body_id);

    // Transform palm rotation Jacobian to body frame
    Eigen::Matrix3d R_palm = palm_q_cur.toRotationMatrix();
    Eigen::MatrixXd J_palm_rot_B = R_palm.transpose() * J_palm_rot_W;

    // Finger Jacobians (in palm frame)
    Eigen::MatrixXd J_thumb = kinematics.computeJacobian(m, d, Body::THUMB_EE, Body::PALM).topRows(3);
    Eigen::MatrixXd J_index = kinematics.computeJacobian(m, d, Body::INDEX_EE, Body::PALM).topRows(3);
    Eigen::MatrixXd J_middle = kinematics.computeJacobian(m, d, Body::MIDDLE_EE, Body::PALM).topRows(3);

    // Palm angular velocity in world frame for orientation control
    Eigen::Vector3d palm_omega_curr_W = states.getBodyAngVel_meas_W(m, d, Body::PALM);
    Eigen::Vector3d thumb_vel = J_thumb * dq;
    Eigen::Vector3d index_vel = J_index * dq;
    Eigen::Vector3d middle_vel = J_middle * dq;

    // Compute desired task-space accelerations (in world frame)
    Eigen::Vector3d palm_orient_acc_des = computePalmOrientAcceleration(
        m, d, palm_q_des, palm_q_cur, palm_omega_curr_W);

    Eigen::Vector3d thumb_acc_des = computeFingerPosAcceleration(
        thumb_des.translation(), thumb_cur.translation(), thumb_vel);

    Eigen::Vector3d index_acc_des = computeFingerPosAcceleration(
        index_des.translation(), index_cur.translation(), index_vel);

    Eigen::Vector3d middle_acc_des = computeFingerPosAcceleration(
        middle_des.translation(), middle_cur.translation(), middle_vel);

    // Build stacked Jacobian and desired acceleration

    // Total task-space dimension: 3 (palm orient) + 3 (thumb) + 3 (index) + 3 (middle) = 12
    int task_dim = 12;
    Eigen::MatrixXd J_stacked(task_dim, nv);
    Eigen::VectorXd xdd_des(task_dim);
    Eigen::VectorXd task_weights(task_dim);

    // Stack Jacobians and desired accelerations with weights
    int row = 0;

    // Palm orientation (world frame) - use world-frame Jacobian for stable control
    J_stacked.block(row, 0, 3, nv) = std::sqrt(w_palm_orient_) * J_palm_rot_W;
    xdd_des.segment(row, 3) = std::sqrt(w_palm_orient_) * palm_orient_acc_des;
    row += 3;

    // Thumb
    J_stacked.block(row, 0, 3, nv) = std::sqrt(w_thumb_) * J_thumb;
    xdd_des.segment(row, 3) = std::sqrt(w_thumb_) * thumb_acc_des;
    row += 3;

    // Index
    J_stacked.block(row, 0, 3, nv) = std::sqrt(w_index_) * J_index;
    xdd_des.segment(row, 3) = std::sqrt(w_index_) * index_acc_des;
    row += 3;

    // Middle
    J_stacked.block(row, 0, 3, nv) = std::sqrt(w_middle_) * J_middle;
    xdd_des.segment(row, 3) = std::sqrt(w_middle_) * middle_acc_des;

    // Null-Space Manipulability Optimization
    // Check palm Jacobian condition number
    double palm_cond = kinematics.checkJacobianCondition(J_palm_rot_W, 1e10);

    // Compute null-space projection to push away from singularities
    Eigen::MatrixXd J_palm_pinv = kinematics.computeJacobianInv(J_palm_rot_W, 0.00, true);
    Eigen::MatrixXd N_palm = Eigen::MatrixXd::Identity(nv, nv) - J_palm_pinv * J_palm_rot_W;

    // Null-space objective: move toward center of joint range
    Eigen::VectorXd q_null(nv);
    for (int i = 0; i < nv; i++) {
        double q_min = m->jnt_range[2*i];
        double q_max = m->jnt_range[2*i + 1];
        double q_mid = (q_min + q_max) / 2.0;
        double q_current = d->qpos[i];
        q_null[i] = q_mid - q_current;
    }

    // Null-space motion (only affects redundant DOFs)
    Eigen::VectorXd qdd_null = N_palm * q_null;

    // Adaptive null-space weight based on conditioning
    double w_null = 1.0;
    if (palm_cond > 1e3) {
        w_null *= std::min(10.0, palm_cond / 1e3);
    }

    // Coupled Joint Constraints
    // Index finger: qdd[5] (JI_L3) = coupling_ratio * qdd[4] (JI_L2)
    // Middle finger: qdd[8] (JP_L3) = coupling_ratio * qdd[7] (JP_L2)
    constexpr double coupling_ratio = 1.0;
    Eigen::MatrixXd A_eq(2, nv);
    A_eq.setZero();
    A_eq(0, 4) = -coupling_ratio;  // JI_L2
    A_eq(0, 5) = 1.0;              // JI_L3
    A_eq(1, 7) = -coupling_ratio;  // JP_L2
    A_eq(1, 8) = 1.0;              // JP_L3

    // Add Palm Velocity/Acceleration Constraints (Prevent Instability)
    // Palm joints are indices 0, 1, 2 (RX, RY, RZ rotations)
    Eigen::VectorXd qdd_palm_limit = Eigen::VectorXd::Zero(nv);

    for (int i = 0; i < 3; i++) {  // Palm joints only
        double q_current = d->qpos[i];
        double qd_current = d->qvel[i];

        // Limit velocity: if |qd| > max_vel, add decelerating acceleration
        if (std::abs(qd_current) > max_palm_velocity) {
            qdd_palm_limit[i] = -50.0 * qd_current;  // Strong damping
        }
    }

    // Self-Collision Avoidance Constraints
    // Inequality constraints: qdd * dt + qd * dt < q_max - q_current
    Eigen::VectorXd qdd_collision_avoidance = Eigen::VectorXd::Zero(nv);

    for (int i = 0; i < nv; i++) {
        double q_min = m->jnt_range[2*i];
        double q_max = m->jnt_range[2*i + 1];
        double q_current = d->qpos[i];

        // Add repulsive acceleration near limits
        if (q_current < q_min + collision_margin) {
            qdd_collision_avoidance[i] = repulsion_gain * (q_min + collision_margin - q_current);
        } else if (q_current > q_max - collision_margin) {
            qdd_collision_avoidance[i] = repulsion_gain * (q_max - collision_margin - q_current);
        }
    }

    // Solve Constrained QP
    /**
     * min   ||J_stacked * qdd - xdd_des||^2 
     *       + w_reg * ||qdd||^2 
     *       + w_null * ||qdd - qdd_null||^2
     *       + w_collision * ||qdd - qdd_collision_avoidance||^2
     *       + w_palm_limit * ||qdd - qdd_palm_limit||^2
     * subject to: A_eq * qdd = 0  (coupling constraints)
     *
     * Using penalty method: add coupling constraint as soft constraint with high weight
     */



    // Augmented cost: H_aug = H + w_coupling * A_eq^T * A_eq + w_collision * I + w_palm_limit * I
    Eigen::MatrixXd H = J_stacked.transpose() * J_stacked +
                        (w_regularization_ + w_null + w_collision + w_palm_limit) * Eigen::MatrixXd::Identity(nv, nv) +
                        w_coupling * A_eq.transpose() * A_eq;

    Eigen::VectorXd g = J_stacked.transpose() * xdd_des +
                        w_null * qdd_null +
                        w_collision * qdd_collision_avoidance +
                        w_palm_limit * qdd_palm_limit;

    Eigen::VectorXd qdd = H.ldlt().solve(g);

    // Hard clamp palm accelerations to ensure safety
    for (int i = 0; i < 3; i++) {
        qdd[i] = std::max(-max_palm_acceleration, std::min(max_palm_acceleration, qdd[i]));
    }

    // Compute torques from dynamics

    // Get mass matrix
    Eigen::MatrixXd M(nv, nv);
    std::vector<double> Mbuf(nv * nv);
    mj_fullM(m, Mbuf.data(), d->qM);
    for (int i = 0; i < nv; i++)
        for (int j = 0; j < nv; j++)
            M(i, j) = Mbuf[i * nv + j];

    // Get Coriolis + gravity
    Eigen::VectorXd C_g(nv);
    for (int i = 0; i < nv; i++)
        C_g[i] = d->qfrc_bias[i];

    // Compute torques: tau = M*qdd + C + G
    Eigen::VectorXd tau = M * qdd + C_g;

    // Apply soft torque limits
    double max_torque = 30.0;
    for (int i = 0; i < nv; i++) {
        if (std::abs(tau[i]) > max_torque) {
            tau[i] = max_torque * std::tanh(tau[i] / max_torque);
        }
    }

    return tau;
}

Eigen::Vector3d OSCControl::computePalmOrientAcceleration(
    const mjModel* m, mjData* d,
    const Eigen::Quaterniond& q_des,
    const Eigen::Quaterniond& q_cur,
    const Eigen::Vector3d& omega_cur
)
{
    // Smooth ramp-up to prevent initial instability
    static bool initialized = false;
    static Eigen::Quaterniond q_initial;
    static double init_time = 0.0;
    const double ramp_duration = 3.0; // 3 second ramp

    if (!initialized) {
        q_initial = q_cur;
        init_time = d->time;
        initialized = true;
    }

    // Compute smooth blend factor with cubic easing
    double elapsed = d->time - init_time;
    double alpha = std::min(1.0, elapsed / ramp_duration);
    alpha = 3.0 * alpha * alpha - 2.0 * alpha * alpha * alpha;

    // Blend from initial to desired orientation
    Eigen::Quaterniond q_des_blended = q_initial.slerp(alpha, q_des);

    Eigen::Vector3d orient_err = computeOrientationError(q_des_blended, q_cur);

    // Clamp orientation error to prevent large corrections
    const double max_orient_err = 0.15;  // rad
    if (orient_err.norm() > max_orient_err) {
        orient_err = orient_err.normalized() * max_orient_err;
    }

    return kp_palm_orient_ * orient_err - kd_palm_orient_ * omega_cur;
}

Eigen::Vector3d OSCControl::computeFingerPosAcceleration(
    const Eigen::Vector3d& pos_des,
    const Eigen::Vector3d& pos_cur,
    const Eigen::Vector3d& vel_cur
)
{
    Eigen::Vector3d pos_err = pos_des - pos_cur;
    return kp_finger_ * pos_err - kd_finger_ * vel_cur;
}

Eigen::Vector3d OSCControl::computeOrientationError(
    const Eigen::Quaterniond& q_des,
    const Eigen::Quaterniond& q_cur
)
{
    // Handle quaternion double-cover: ensure shortest path
    Eigen::Quaterniond q_des_corrected = q_des;
    if (q_des.coeffs().dot(q_cur.coeffs()) < 0.0) {
        q_des_corrected.coeffs() = -q_des.coeffs();
    }

    // Compute rotation error in world frame for stable orientation control
    Eigen::Matrix3d R_cur = q_cur.toRotationMatrix();
    Eigen::Matrix3d R_des = q_des_corrected.toRotationMatrix();
    Eigen::Matrix3d R_error = R_des * R_cur.transpose();

    // Convert to axis-angle
    Eigen::AngleAxisd aa(R_error);
    return aa.angle() * aa.axis();
}

Eigen::Vector3d OSCControl::computePalmTorquesFromFingers(
    const mjModel* m,
    mjData* d,
    const Eigen::VectorXd& tau_whole_hand
)
{
    // Compute reaction torques on palm from finger dynamics
    // The palm is treated as a rigid body with fingers attached
    //
    // Key insight: The mass matrix couples palm and finger DOFs
    // M = [ M_pp  M_pf ]
    //     [ M_fp  M_ff ]
    //
    // Dynamics: M*qdd + C = tau
    // For palm DOFs: M_pp*qdd_p + M_pf*qdd_f + C_p = tau_p
    //
    // The term M_pf*qdd_f represents the reaction torques on palm from finger accelerations

    int nv = m->nv;

    // Get full mass matrix (Row Major)
    Eigen::MatrixXd M(nv, nv);
    std::vector<double> Mbuf(nv * nv);
    mj_fullM(m, Mbuf.data(), d->qM);
    for (int i = 0; i < nv; i++)
        for (int j = 0; j < nv; j++)
            M(i, j) = Mbuf[i * nv + j];

    // Get bias forces (Coriolis + gravity)
    Eigen::VectorXd C_g(nv);
    for (int i = 0; i < nv; i++)
        C_g[i] = d->qfrc_bias[i];

    // Extract finger torques from whole hand solution
    Eigen::VectorXd tau_fingers = tau_whole_hand.tail(nv - PALM_DOF);

    // Compute finger accelerations using inverse dynamics
    // For fingers: M_ff*qdd_f + M_fp*qdd_p + C_f = tau_f
    // Approximate: qdd_f ≈ inv(M_ff) * (tau_f - C_f) assuming M_fp*qdd_p is small
    Eigen::MatrixXd M_ff = M.bottomRightCorner(nv - PALM_DOF, nv - PALM_DOF);
    Eigen::VectorXd C_f = C_g.tail(nv - PALM_DOF);

    // Solve for finger accelerations
    Eigen::VectorXd qdd_fingers = M_ff.ldlt().solve(tau_fingers - C_f);

    // Compute coupling forces: M_pf * qdd_f
    Eigen::MatrixXd M_pf = M.topRightCorner(PALM_DOF, nv - PALM_DOF);
    Eigen::Vector3d tau_coupling = M_pf * qdd_fingers;

    // Transform to palm body frame
    Eigen::Quaterniond q_palm = states.getBodyQuat_meas_W(m, Body::PALM);
    Eigen::Matrix3d R_palm = q_palm.toRotationMatrix();
    Eigen::Vector3d tau_palm = R_palm.transpose() * tau_coupling;

    return tau_palm;
}
