#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <Eigen/Geometry>
#include <mujoco/mujoco.h>

#include "mujoco_driver.hpp"
#include "mujoco_api.hpp"
#include "operational_space_control.hpp"
#include "attitude_PD_control.hpp"
#include "model_predictive_control.hpp"
#include "kinematics.hpp"
#include "parse_data.hpp"
#include "states.hpp"
#include "body_enum.hpp"
#include "state_logger.hpp"


int main(int argc, char** argv)
{
    // Path to model
    std::filesystem::path model_path = "../model/7DHand.xml";

    // Create the MuJoCo driver (window + driver + snapshot)
    mj::MuJoCoDriver driver;

    // Include the controller & data parser
    Kinematics kinematics;
    States states;
    ParseData csvData;
    OSCControl osc(kinematics, states, csvData);
    PDControl pd(kinematics, states, csvData);
    MPCControl mpc(kinematics, states, csvData);

    // Initialize state logger
    StateLogger logger;
    logger.initialize("../logs/state_tracking.csv", 10);  // Log every 10 control steps

    // --- Load the XML model ---
    std::string error;
    if (!driver.loadModel(model_path, error))
    {
        std::cerr << "Failed to load model: " << model_path << "\n";
        std::cerr << "MuJoCo error: " << error << "\n";
        return 1;
    }

    // --- Open the rendering window ---
    if (!driver.openWindow(1200, 900, "7D Hand driverulation"))
    {
        std::cerr << "Failed to open MuJoCo window\n";
        return 1;
    }

    // Tell global context to use this driver & window
    driver.setActive();

    // Load the CSV File for reference positions (base and end effectors positions and orientations)
    csvData.loadCSV("../logs/hand_tracking_data_pinch.csv");

    // --- Add a control callback ---
    int64_t time_start = 0;

    mj::mjSetControlCallback([&driver, &csvData, &states, &kinematics, &osc, &mpc, &pd, &time_start, &logger](const mjModel* m, mjData* d) {

        // --- Initialize states ---
        if (!states.isInitialized()){
            states.initializeStorage(m);
            time_start = static_cast<int64_t>(d->time * 1000.0);  // Convert seconds to milliseconds
        }
        states.updateStates(m, d);

        // --- Control index ---
        int64_t curr_time_ms = static_cast<int64_t>(d->time * 1000.0);  // Convert seconds to milliseconds
        int idx = csvData.getRefIdx(time_start, curr_time_ms, csvData.getTimestamps());

        // --- Operational Space Control ---
        // [roll, pitch, yaw, fingers]
        Eigen::VectorXd tau = osc.computeJointTorques(m, d, idx);

        // --- Compute external torques from fingers ---
        // Eigen::Vector3d tau_ext = osc.computePalmTorquesFromFingers(m, d, tau);

        // --- Palm MPC Control ---
        // Eigen::Vector3d tau_palm = mpc.computePalmTorques(m, d, tau_ext, idx, 5);    
        // tau.head(3) = tau_palm;

        // --- Palm Impeadance Control ---
        Eigen::Vector3d tau_ext(0, 0, 0); 
        Eigen::Vector3d tau_palm2 = pd.computePalmTorques(m, d, Body::PALM, tau_ext, idx);
        tau.head(3) += tau_palm2;

        // --- Log actual vs desired states ---
        logger.logCompleteState(m, d, states, csvData, idx);

        // --- Torque ramp-up to prevent instability ---
        // static const double ramp_dur = 0.5;  // Ramp up over 2 seconds
        // double ramp_factor = 0.0;
        // if (d->time < ramp_dur) {
        //     // Cubic ramp function: smooth from 0 to 1
        //     double t_norm = d->time / ramp_dur;
        //     ramp_factor = 3.0 * t_norm * t_norm - 2.0 * t_norm * t_norm * t_norm;
        //     tau *= ramp_factor;
        // }

        // --- Apply torques to actuators ---
        for (int j = 0; j < m->nu; j++) {
            mjtNum torque = static_cast<mjtNum>(tau[j]);           
            d->ctrl[j] = torque;
        }

    });

    // --- driverulation loop ---
    const mjtNum driver_dt = 0.001;  // dt
    const mjtNum display_dt = 0.01;  // render time

    while (!driver.shouldCloseWindow())
    {
        // Advance driverulation
        driver.step(display_dt, driver_dt);

        // Render scene
        driver.render();
    }

    // --- Close window cleanly ---
    driver.closeWindow();

    // --- Close logger ---
    logger.close();
    std::cout << "State tracking log saved to ../logs/state_tracking.csv\n";

    return 0;
}
