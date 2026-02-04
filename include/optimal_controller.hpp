#pragma once
#include <Eigen/Dense>
#include <mujoco/mujoco.h>
#include "body_enum.hpp"
#include "states.hpp"
#include "parse_data.hpp"
#include "kinematics.hpp"

/**
 * Optimal controller using iterative LQR (iLQR) for finger tracking
 * Computes optimal joint torques and desired joint positions to minimize tracking error
 */
class OptimalController {
private:
    Kinematics& kinematics;
    States& states;
    ParseData& csvData;

    // iLQR parameters
    int horizon_;           // Planning horizon (timesteps)
    double dt_;              // Timestep duration
    double alpha_;           // Line search parameter
    int max_iterations_;    // Max iLQR iterations

    // Cost weights
    double Q_pos_;           // Position tracking weight
    double Q_vel_;           // Velocity regularization weight
    double R_torque_;        // Torque regularization weight

public:
    OptimalController(Kinematics& kin, States& st, ParseData& csv,
                      int horizon = 10, double dt = 0.000)
        : kinematics(kin), states(st), csvData(csv),
          horizon_(horizon), dt_(dt), alpha_(0.0), max_iterations_(5),
          Q_pos_(100.0), Q_vel_(1.0), R_torque_(0.00) {}

    /**
     * Compute optimal control for a single finger using iLQR
     * Returns: optimal joint torques for current timestep
     */
    Eigen::VectorXd computeOptimalControl(
        const mjModel* m, mjData* d,
        int csv_idx,
        Body finger_body,
        Body base_body
    );

private:
    /**
     * Forward rollout dynamics using current control sequence
     */
    void forwardRollout(
        const mjModel* m, mjData* d,
        const std::vector<Eigen::VectorXd>& u_sequence,
        std::vector<Eigen::VectorXd>& x_sequence,
        Body finger_body, Body base_body
    );

    /**
     * Backward pass: compute optimal gains
     */
    void backwardPass(
        const mjModel* m, mjData* d,
        const std::vector<Eigen::VectorXd>& x_sequence,
        const std::vector<Eigen::VectorXd>& u_sequence,
        std::vector<Eigen::MatrixXd>& K_sequence,
        std::vector<Eigen::VectorXd>& k_sequence,
        Body finger_body, Body base_body,
        int csv_idx
    );

    /**
     * Compute linearized dynamics: dx_{t+1} = A*dx_t + B*du_t
     */
    void linearizeDynamics(
        const mjModel* m, mjData* d,
        const Eigen::VectorXd& x, const Eigen::VectorXd& u,
        Eigen::MatrixXd& A, Eigen::MatrixXd& B,
        Body finger_body, Body base_body
    );

    /**
     * Compute cost for a given state and control
     */
    double computeCost(
        const mjModel* m,
        const Eigen::VectorXd& x,
        const Eigen::VectorXd& u,
        const Eigen::Vector3d& target_pos,
        Body finger_body, Body base_body
    );
};
