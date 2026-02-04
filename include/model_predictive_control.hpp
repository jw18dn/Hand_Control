#pragma once
#include <Eigen/Dense>
#include <mujoco/mujoco.h>
#include <vector>
#include "body_enum.hpp"
#include "states.hpp"
#include "parse_data.hpp"
#include "kinematics.hpp"


class MPCControl {
private:
    Kinematics& kinematics;
    States& states;
    ParseData& csvData;

    Eigen::Vector3d saturate(const Eigen::Vector3d &v, double limit);


public:
    MPCControl(Kinematics& kin, States& st, ParseData& csv): kinematics(kin), states(st), csvData(csv) {}
    /**
     * Model Predictive Controller for palm orientation tracking
     * Optimizes torque sequence over N-step horizon to track quaternion trajectory
     *
     * @param m MuJoCo model
     * @param d MuJoCo data (current state)
     * @param tau_ext External torque on palm (fixed for horizon)
     * @param csv_idx Current CSV index (fixed for horizon)
     * @param N Prediction horizon length
     * @return Optimal palm torque for current timestep
     */
    Eigen::Vector3d computePalmTorques(
        const mjModel* m,
        mjData* d,
        const Eigen::Vector3d& tau_ext,
        int csv_idx,
        int N = 10
    );

private:

};
