#include "attitude_PD_control.hpp"
#include "mujoco_api.hpp"
#include <iostream>
#include <Eigen/Sparse>

Eigen::Vector3d PDControl::computePalmTorques(
    const mjModel* m, 
    mjData* d, 
    const Body& body_enum, 
    Eigen::VectorXd tau_ext, 
    int csv_idx) 
{
    double kp_orient = 0.1;   
    double kd_orient = 0.15; 

    // Get body ID
    int body_id = getBodyRef(m, body_enum);

    // Get current orientation (world frame)
    Eigen::Matrix3d R_cur = states.getBodyRot_meas_W(m, body_enum);

    // Get desired orientation (world frame)
    Eigen::Matrix3d R_des = csvData.getBodyIso_des_W(csv_idx, body_enum).linear();

    // Compute orientation error in body frame: R_error = R_cur^T * R_des
    Eigen::Matrix3d R_error = R_cur.transpose() * R_des;

    // Convert rotation matrix to axis-angle
    Eigen::AngleAxisd angle_axis(R_error);

    // Rotation vector (axis * angle) in body frame
    Eigen::Vector3d orient_error = angle_axis.axis() * angle_axis.angle();
    double angle = angle_axis.angle();
    if (angle < 1e-8) {
        orient_error.setZero();
    } else {
        orient_error = angle_axis.axis() * angle;
    }

    // Get current angular velocity in body frame using MuJoCo
    Eigen::Vector3d omega_cur_B = states.getBodyAngVel_meas_B(m, d, body_enum);

    // Get desired angular velocity (already in body frame from ParseData)
    Eigen::Vector3d omega_des_B = csvData.getBodyAngularVel_des_B(csv_idx, body_enum);

    // Get desired angular acceleration (already in body frame)
    Eigen::Vector3d alpha_des_B = csvData.getBodyAngularAcc_des_B(csv_idx, body_enum);

    // Angular velocity error (both in body frame)
    Eigen::Vector3d omega_error = omega_cur_B - omega_des_B;

    // PD control (correct signs: tau = kp*error + kd*error_dot)
    // Use only the angular (bottom 3 rows) part of the Jacobian for orientation control
    Eigen::MatrixXd J_palm_W = kinematics.computeJacobian(m, d, Body::PALM, Body::UNIVERSE);
    Eigen::MatrixXd J_palm_rot_W = J_palm_W.block(3, 0, 3, 3);  // Extract angular part (3xN)
    Eigen::Vector3d tau_pd = (kp_orient * orient_error - kd_orient * omega_error);

    // Extract the 3x3 inertia matrix from cinert (in body frame)
    const mjtNum* cinert = d->cinert + 10 * body_id;
    Eigen::Matrix3d I;
    I(0,0) = cinert[4];  // I_xx
    I(1,1) = cinert[5];  // I_yy
    I(2,2) = cinert[6];  // I_zz

    // Feedforward acceleration term (with ramp)
    // tau_ff = I * (alpha_des - omega_cur x omega_des)
    Eigen::Vector3d omega_cross_omega_des = omega_cur_B.cross(omega_des_B);
    Eigen::Vector3d tau_ff = I * (alpha_des_B - omega_cross_omega_des);

    // Coriolis/centrifugal compensation
    // tau_coriolis = omega_cur x (I * omega_cur)
    Eigen::Vector3d tau_coriolis = omega_cur_B.cross(I * omega_cur_B);

    // Total control torque (subtract external torques for compensation)
    // Eigen::Vector3d tau = tau_pd + tau_ff + tau_coriolis - tau_ext.head(3);
    Eigen::Vector3d tau = tau_pd;

    return tau;
}
