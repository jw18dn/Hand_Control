#pragma once
#include <Eigen/Dense>
#include <mujoco/mujoco.h>
#include <vector>
#include "body_enum.hpp"
#include "states.hpp"
#include "parse_data.hpp"
#include "kinematics.hpp"


class PDControl {
private:
    Kinematics& kinematics;
    States& states;
    ParseData& csvData;

public:
    PDControl(Kinematics& kin, States& st, ParseData& csv): kinematics(kin), states(st), csvData(csv) {}

    /**
     * Geometric attitude controller using SO(3) error formulation
     * Implements a PD controller on SO(3) with feedforward and external torque compensation
     */
    Eigen::Vector3d computePalmTorques(
        const mjModel* m,
        mjData* d,
        const Body& body_enum,
        Eigen::VectorXd tau_ext,
        int csv_idx
    );

private:

};
