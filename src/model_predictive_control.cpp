#include "model_predictive_control.hpp"
#include "mujoco_api.hpp"
#include <iostream>
#include <Eigen/Sparse>

/**
 * Component-wise saturation of a 3D vector
 * @param v Input vector
 * @param limit Saturation limit (applied symmetrically)
 * @return Saturated vector
 */
Eigen::Vector3d MPCControl::saturate(const Eigen::Vector3d &v, double limit) {
    if (!std::isfinite(limit) || limit <= 0.0) return v;
    Eigen::Vector3d out = v;
    for (int i = 0; i < 3; ++i) {
        out(i) = std::max(-limit, std::min(limit, out(i)));
    }
    return out;
}

Eigen::Vector3d MPCControl::computePalmTorques(
    const mjModel* m,
    mjData* d,
    const Eigen::Vector3d& tau_ext,
    int csv_idx,
    int N
)
{
    /**
     * Model Predictive Controller for palm orientation tracking
     */
    const int body_id = getBodyRef(m, Body::PALM);
    const double u_max = 10.0;
    const double dt = m->opt.timestep;

    // Cost weights
    const double Q_orient = 100.0;  // Orientation tracking weight
    const double Q_omega = 200.0;     // Angular velocity tracking weight
    const double R = 0.01;           // Control effort weight

    // Get current state
    const mjtNum* quat_cur_data = d->xquat + 4 * body_id;
    Eigen::Quaterniond q_cur(quat_cur_data[0], quat_cur_data[1],
                             quat_cur_data[2], quat_cur_data[3]);

    // TODO what frame is this in                             
    Eigen::Vector3d omega_0 = states.getBodyAngVel_meas_B(m, d, Body::PALM);

    // Get composite inertia
    const mjtNum* cinert = d->cinert + 10 * body_id;
    Eigen::Matrix3d I_palm;
    I_palm(0,0) = cinert[4];
    I_palm(1,1) = cinert[5];
    I_palm(2,2) = cinert[6];
    I_palm(0,1) = I_palm(1,0) = cinert[7];
    I_palm(0,2) = I_palm(2,0) = cinert[8];
    I_palm(1,2) = I_palm(2,1) = cinert[9];
    Eigen::Matrix3d I_inv = I_palm.inverse();

    // Load reference trajectory
    std::vector<Eigen::Quaterniond> q_ref(N+1);
    std::vector<Eigen::Vector3d> omega_ref(N+1);

    q_ref[0] = q_cur;
    for (int k = 0; k <= N; k++) {
        double future_time = d->time + k * dt;

        if (k > 0) {
            q_ref[k] = csvData.getBodyQuat_des_W_at(d->time, future_time, Body::PALM);
        }
        omega_ref[k] = csvData.getBodyAngularVel_des_B_at(d->time, future_time, Body::PALM);
    }

    // Decision variables: z = [omega_1, ..., omega_N, u_0, ..., u_{N-1}]
    // Dimension: 3*N (omega) + 3*N (u) = 6*N
    int n_vars = 6 * N;

    // Build QP matrices: min 0.5 * z^T * H * z + g^T * z
    //                    s.t. A_eq * z = b_eq (dynamics constraints)

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_vars, n_vars);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(n_vars);

    // Dynamics constraint matrix: omega_{k+1} = omega_k + dt * I_inv * (u_k - omega_k × I*omega_k + tau_ext)
    // Linearized: omega_{k+1} ≈ omega_k + dt * I_inv * u_k + dt * I_inv * tau_ext (ignoring Coriolis for now)
    Eigen::MatrixXd A_eq = Eigen::MatrixXd::Zero(3*N, n_vars);
    Eigen::VectorXd b_eq = Eigen::VectorXd::Zero(3*N);

    // Predict orientation along trajectory for cost computation
    std::vector<Eigen::Quaterniond> q_pred(N+1);
    q_pred[0] = q_cur;

    for (int k = 0; k < N; k++) {
        // Indices in decision variable vector
        int omega_idx_next = 3 * k;      // omega_{k+1} starts at 3*k
        int omega_idx_prev = 3 * (k-1);  // omega_k starts at 3*(k-1)
        int u_idx = 3 * N + 3 * k;       // u_k starts at 3*N + 3*k

        // === Dynamics constraints ===
        // omega_{k+1} = omega_k + dt * I_inv * (u_k + tau_ext - omega_k × I*omega_k)
        // Linearized around reference: omega_{k+1} - omega_k - dt*I_inv*u_k = dt*I_inv*(tau_ext - omega_ref_k × I*omega_ref_k)

        Eigen::Vector3d omega_ref_k = (k == 0) ? omega_0 : omega_ref[k];  // Reference for linearization
        Eigen::Vector3d coriolis = omega_ref_k.cross(I_palm * omega_ref_k);

        // Constraint: omega_{k+1} - omega_k - dt*I_inv*u_k = dt*I_inv*(tau_ext - coriolis)

        // omega_{k+1} coefficient: +I
        A_eq.block<3,3>(3*k, omega_idx_next) = Eigen::Matrix3d::Identity();

        // omega_k coefficient: -I (only if k > 0, otherwise it's the initial condition)
        if (k > 0) {
            A_eq.block<3,3>(3*k, omega_idx_prev) = -Eigen::Matrix3d::Identity();
        }

        // u_k coefficient: -dt*I_inv
        A_eq.block<3,3>(3*k, u_idx) = -dt * I_inv;

        // RHS
        if (k == 0) {
            // omega_1 - dt*I_inv*u_0 = omega_0 + dt*I_inv*(tau_ext - coriolis)
            b_eq.segment<3>(3*k) = omega_0 + dt * I_inv * (tau_ext - coriolis);
        } else {
            // omega_{k+1} - omega_k - dt*I_inv*u_k = dt*I_inv*(tau_ext - coriolis)
            b_eq.segment<3>(3*k) = dt * I_inv * (tau_ext - coriolis);
        }

        // === Cost function ===

        // Predict orientation using current best estimate
        Eigen::Vector3d omega_for_pred = omega_ref[k+1];  // Use reference for prediction

        Eigen::Vector3d theta = dt * omega_for_pred;
        double angle = theta.norm();
        Eigen::Quaterniond dq;
        if (angle < 1e-8) {
            dq = Eigen::Quaterniond::Identity();
        } else {
            dq = Eigen::Quaterniond(Eigen::AngleAxisd(angle, theta/angle));
        }
        q_pred[k+1] = q_pred[k] * dq;

        // Orientation error at k+1
        Eigen::Matrix3d R_pred = q_pred[k+1].toRotationMatrix();
        Eigen::Matrix3d R_ref = q_ref[k+1].toRotationMatrix();
        Eigen::Matrix3d R_error = R_pred.transpose() * R_ref;

        Eigen::AngleAxisd angle_axis(R_error);
        Eigen::Vector3d e_orient = angle_axis.axis() * angle_axis.angle();
        if (angle_axis.angle() < 1e-8) {
            e_orient.setZero();
        }

        // Angular velocity cost: Q_omega * ||omega_{k+1} - omega_ref_{k+1}||^2
        // Expanded: Q_omega * (omega^T*omega - 2*omega_ref^T*omega + const)
        // QP form: min 0.5*x^T*H*x + g^T*x
        // So: 0.5*omega^T*(2*Q_omega*I)*omega + (-2*Q_omega*omega_ref)^T*omega
        H.block<3,3>(omega_idx_next, omega_idx_next) += 2.0 * Q_omega * Eigen::Matrix3d::Identity();
        g.segment<3>(omega_idx_next) -= 2.0 * Q_omega * omega_ref[k+1];

        // Control cost: R * ||u_k||^2
        // Expanded: R * u^T * u
        // QP form: 0.5*u^T*(2*R*I)*u
        H.block<3,3>(u_idx, u_idx) += 2.0 * R * Eigen::Matrix3d::Identity();

        // Orientation cost (linearized gradient)
        // The orientation error depends on omega_{k+1} through integration
        // Gradient approximation: ∂||e_orient||^2/∂omega ≈ 2*dt*e_orient
        g.segment<3>(omega_idx_next) += 2.0 * Q_orient * dt * e_orient;
    }

    // Solve constrained QP with penalty method
    // Convert equality constraints to penalty: min 0.5*z^T*H*z + g^T*z + 0.5*λ||A_eq*z - b_eq||^2
    double lambda = 1e6;  // Large penalty weight for constraint enforcement

    Eigen::MatrixXd H_aug = H + lambda * (A_eq.transpose() * A_eq);
    Eigen::VectorXd g_aug = g - lambda * (A_eq.transpose() * b_eq);

    // Solve: H_aug * z = -g_aug
    Eigen::VectorXd z_opt = H_aug.ldlt().solve(-g_aug);

    // Extract first control action u_0
    Eigen::Vector3d u_0 = z_opt.segment<3>(3*N);

    // Debug output
    static int counter = 0;
    if (counter % 100 == 0) {
        std::cout << "\n=== MPC Debug (iter " << counter << ") ===" << std::endl;

        // Current vs desired orientation
        Eigen::Matrix3d R_cur = q_cur.toRotationMatrix();
        Eigen::Matrix3d R_des = q_ref[1].toRotationMatrix();
        Eigen::Matrix3d R_error = R_cur.transpose() * R_des;
        Eigen::AngleAxisd aa(R_error);
        std::cout << "Orient error (rad): " << (aa.axis() * aa.angle()).transpose() << std::endl;

        // Current vs desired angular velocity
        std::cout << "omega_cur: " << omega_0.transpose() << std::endl;
        std::cout << "omega_ref: " << omega_ref[0].transpose() << std::endl;

        // Computed control
        std::cout << "u_0 (before sat): " << z_opt.segment<3>(3*N).transpose() << std::endl;
        std::cout << "u_0 (after sat): " << u_0.transpose() << std::endl;
        std::cout << "tau_ext: " << tau_ext.transpose() << std::endl;
        // std::cout << "tau_coriolis: " << tau_coriolis.transpose() << std::endl;

        // First predicted omega
        std::cout << "omega_1 (opt): " << z_opt.segment<3>(0).transpose() << std::endl;

        // Constraint violation
        Eigen::VectorXd constraint_error = A_eq * z_opt - b_eq;
        std::cout << "Constraint violation norm: " << constraint_error.norm() << std::endl;
    }
    counter++;

    // Apply saturation
    u_0 = saturate(u_0, u_max);

    return u_0;
}