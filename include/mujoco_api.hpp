#pragma once

#include <functional>
#include <memory>

#include "GLFW/glfw3.h"
#include "mujoco/mujoco.h"

namespace mj {

struct mjWindowContext
{
    GLFWwindow* window = nullptr; // GLFW window
    struct
    {
        bool button_left = false;
        bool button_middle = false;
        bool button_right = false;
        double lastx = 0;
        double lasty = 0;
    } mouse;            // GLFW mouse data
    mjvCamera camera;   // Camera for rendering
    mjvPerturb perturb; // Perturbation for rendering
    mjvOption option;   // Rendering options
    mjvScene scene;     // Scene for rendering
    mjrContext context; // Rendering context
};

struct mjSimContext
{
    mjModel* model = nullptr; // MuJoCo model
    mjData* data = nullptr;   // MuJoCo data
    std::size_t seq = 0;
};

struct mjSimSnapshot
{
    mjtNum time = 0.0;
    std::array<mjtNum, 3> body_position;
    std::array<mjtNum, 4> body_quaternion;
    std::array<mjtNum, 3> body_linear_velocity;
    std::array<mjtNum, 3> body_angular_velocity;
    std::array<mjtNum, 3> body_linear_acceleration;
    std::array<mjtNum, 3> body_angular_acceleration;
    std::vector<mjtNum> joint_positions;
    std::vector<mjtNum> joint_velocities;
    std::vector<mjtNum> joint_accelerations;
    std::vector<mjtNum> joint_controls;
};

inline auto& _mjWindowContext() noexcept
{
    static std::shared_ptr<mjWindowContext> data = nullptr; // Global GLFW data
    return data;
}

inline auto& _mjSimContext() noexcept
{
    static std::shared_ptr<mjSimContext> data = nullptr; // Global MuJoCo data
    return data;
}

/* Keyboard/Mouse Callbacks */

static inline void _glfwKeyPressCallback(GLFWwindow*, int key, int, int act, int)
{
    if (auto sc = _mjSimContext())
    {
        if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE)
        {
            mj_resetData(sc->model, sc->data);
            mj_forward(sc->model, sc->data);
        }
    }
}

static inline void _glfwMouseClickCallback(GLFWwindow* window, int, int, int)
{
    if (auto wc = _mjWindowContext())
    {
        wc->mouse.button_left = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        wc->mouse.button_middle = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
        wc->mouse.button_right = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        glfwGetCursorPos(window, &wc->mouse.lastx, &wc->mouse.lasty);
    }
}

static inline void _glfwMouseMoveCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (auto wc = _mjWindowContext())
    {
        if (!wc->mouse.button_left && !wc->mouse.button_middle && !wc->mouse.button_right)
            return;

        const double dx = xpos - wc->mouse.lastx;
        const double dy = ypos - wc->mouse.lasty;
        wc->mouse.lastx = xpos;
        wc->mouse.lasty = ypos;

        int width, height;
        glfwGetWindowSize(window, &width, &height);

        bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ||
                         (glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

        mjtMouse action;
        if (wc->mouse.button_right)
        {
            // camera movement
            action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
        }
        else if (wc->mouse.button_left)
        {
            // camera rotation
            action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
        }
        else
        {
            // camera zoom
            action = mjMOUSE_ZOOM;
        }

        if (auto sc = _mjSimContext())
        {
            mjv_moveCamera(sc->model, action, dx / height, dy / height, &wc->scene, &wc->camera);
        }
    }
}

static inline void _glfwMouseScrollCallback(GLFWwindow*, double, double yoffset)
{
    auto wc = _mjWindowContext();
    auto sc = _mjSimContext();
    if (wc && sc)
    {
        mjv_moveCamera(sc->model, mjMOUSE_ZOOM, 0, 0.05 * yoffset, &wc->scene, &wc->camera);
    }
}

static std::function<void(const mjModel* m, mjData* d)> k_mj_control_func;
inline void _mjControlCallback(const mjModel* m, mjData* d)
{
    if (k_mj_control_func)
        k_mj_control_func(m, d);
}

/// @brief Load a model from a MJ XML file
/// @note `error_buf` needs to be properly sized beforehand
inline bool mjLoadModel(mjSimContext* sc, const char* filename, char* error_buf)
{
    sc->model = mj_loadXML(filename, nullptr, error_buf, sizeof(error_buf));
    if (!sc->model)
        return false;

    sc->data = mj_makeData(sc->model);
    if (!sc->data)
    {
        mj_deleteModel(sc->model);
        sc->model = nullptr;
        return false;
    }
    return true;
}

/// @brief Use a custom control callback function
/// @note Gets called from inside `mj_step`
template <class Func> inline void mjSetControlCallback(Func f)
{
    k_mj_control_func = f;
    mjcb_control = _mjControlCallback;
}

/// @brief Free a sim context model and data
inline void mjFreeModel(mjSimContext* sc)
{
    if (sc->data)
    {
        mj_deleteData(sc->data);
        sc->data = nullptr;
    }
    if (sc->model)
    {
        mj_deleteModel(sc->model);
        sc->model = nullptr;
    }
}

/// @brief Open a window for the simulation
inline bool mjOpenWindow(mjSimContext* sc, mjWindowContext* wc, int width, int height, const char* title)
{
    if (!glfwInit())
    {
        return false;
    }
    wc->window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!wc->window)
    {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(wc->window);
    glfwSwapInterval(1);
    glfwSetKeyCallback(wc->window, _glfwKeyPressCallback);
    glfwSetCursorPosCallback(wc->window, _glfwMouseMoveCallback);
    glfwSetMouseButtonCallback(wc->window, _glfwMouseClickCallback);
    glfwSetScrollCallback(wc->window, _glfwMouseScrollCallback);
    mjv_defaultCamera(&wc->camera);
    mjv_defaultPerturb(&wc->perturb);
    mjv_defaultOption(&wc->option);
    mjr_defaultContext(&wc->context);
    mjv_makeScene(sc->model, &wc->scene, 1000);
    mjr_makeContext(sc->model, &wc->context, mjFONTSCALE_150);
    return true;
}

/// @brief Check if the window should close
/// @note Returns `true` if the window doesn't exist
inline bool mjShouldWindowClose(mjWindowContext* wc)
{
    if (!wc->window)
        return true;
    return glfwWindowShouldClose(wc->window);
}

/// @brief Close the sim window
inline void mjCloseWindow(mjWindowContext* wc)
{
    if (wc->window)
    {
        glfwDestroyWindow(wc->window);
        wc->window = nullptr;
        glfwTerminate();
    }
    mjv_freeScene(&wc->scene);
    mjr_freeContext(&wc->context);
    _mjWindowContext() = nullptr;
}

/// @brief Step the simulation for a given duration
inline void mjStep(mjSimContext* sc, mjSimSnapshot* snap, mjtNum duration, mjtNum sim_dt)
{
    sc->model->opt.timestep = sim_dt;

    const auto end_time = sc->data->time + duration;
    while (sc->data->time < end_time)
    {
        mj_step(sc->model, sc->data);
        ++sc->seq;
    }

    snap->time = sc->data->time;
    snap->body_position = {sc->data->qpos[0], sc->data->qpos[1], sc->data->qpos[2]};
    snap->body_quaternion = {sc->data->qpos[3], sc->data->qpos[4], sc->data->qpos[5], sc->data->qpos[6]};
    snap->body_linear_velocity = {sc->data->qvel[0], sc->data->qvel[1], sc->data->qvel[2]};
    snap->body_angular_velocity = {sc->data->qvel[3], sc->data->qvel[4], sc->data->qvel[5]};
    snap->body_linear_acceleration = {sc->data->qacc[0], sc->data->qacc[1], sc->data->qacc[2]};
    snap->body_angular_acceleration = {sc->data->qacc[3], sc->data->qacc[4], sc->data->qacc[5]};
    snap->joint_positions.resize(sc->model->nu);
    snap->joint_velocities.resize(sc->model->nu);
    snap->joint_accelerations.resize(sc->model->nu);
    for (int i = 0; i < sc->model->nu; i++)
    {
        const int jnt_id = sc->model->actuator_trnid[i * 2];
        const int qpos_adr = sc->model->jnt_qposadr[jnt_id];
        const int qvel_adr = sc->model->jnt_dofadr[jnt_id];
        snap->joint_positions[i] = sc->data->qpos[qpos_adr];
        snap->joint_velocities[i] = sc->data->qvel[qvel_adr];
        snap->joint_accelerations[i] = sc->data->qacc[qvel_adr];
    }
    // NOTE: Commented out to allow control callback to set controls
    // If you want to use driver.setJointTorques(), uncomment these lines
    // for (std::size_t i = 0; i < snap->joint_controls.size(); ++i)
    // {
    //     sc->data->ctrl[i] = snap->joint_controls[i];
    // }
}

/// @brief Render the sim in the window
inline void mjRender(mjSimContext* sc, mjWindowContext* wc)
{
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(wc->window, &viewport.width, &viewport.height);

    mjv_updateScene(sc->model, sc->data, &wc->option, &wc->perturb, &wc->camera, mjCAT_ALL, &wc->scene);
    mjr_render(viewport, &wc->scene, &wc->context);

    glfwSwapBuffers(wc->window);
    glfwPollEvents();
}

/// @brief Reset the sim environment
/// @note Use this rather than closing/reopening window
inline void mjResetEnvironment(mjSimContext* sc)
{
    mj_resetData(sc->model, sc->data);
    mj_forward(sc->model, sc->data);
}
} // namespace mj
