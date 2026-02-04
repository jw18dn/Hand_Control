#pragma once
#include <Eigen/Dense>
#include <mujoco/mujoco.h>
#include "body_enum.hpp"

// Should merge the data parser in here as well

class States {
private:
    bool initialized_ = false;
    Eigen::VectorXd q_meas_W_;  // Generalized Positions (size nq)
    Eigen::VectorXd dq_meas_W_; // Generalized Velocities (size nv)
    std::vector<Eigen::Quaterniond> body_quat_;
    std::vector<Eigen::Isometry3d> body_iso_;

public:

    void updateStates(const mjModel* m, const mjData* d);
    void initializeStorage(const mjModel* m);
    bool isInitialized() const { return initialized_; }

    // Body Position & Orientation (these are wrong)
    const Eigen::Matrix3d getBodyRot_meas_W(const mjModel* m, const Body body) const {return body_iso_[getBodyRef(m, body)].linear();}
    const Eigen::Vector3d getBodyPos_meas_W(const mjModel* m, const Body body) const {return body_iso_[getBodyRef(m, body)].translation();}
    const Eigen::Isometry3d getBodyIso_meas_W(const mjModel* m, const Body body) const {return body_iso_[getBodyRef(m, body)];}
    const Eigen::Quaterniond getBodyQuat_meas_W(const mjModel* m, const Body body) const {return body_quat_[getBodyRef(m, body)];}

    // Body Velocities
    const Eigen::Vector3d getBodyAngVel_meas_W(const mjModel* m, const mjData* d, const Body body);
    const Eigen::Vector3d getBodyTransVel_meas_W(const mjModel* m, const mjData* d, const Body body);
    const Eigen::Vector3d getBodyAngVel_meas_B(const mjModel* m, const mjData* d, const Body body);
    const Eigen::Vector3d getBodyTransVel_meas_B(const mjModel* m, const mjData* d, const Body body);

    // Getters
    const Eigen::VectorXd get_q_meas_W() const {return q_meas_W_;}
    const Eigen::VectorXd get_dq_meas_W() const {return dq_meas_W_;}
};
