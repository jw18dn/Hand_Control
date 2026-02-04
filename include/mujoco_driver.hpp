#pragma once

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <memory>
#include <ranges>
#include <span>

#include "mujoco_api.hpp"

namespace mj {

/// @brief MuJoCo Driver object
class MuJoCoDriver final
{
  public:
    /// @brief Create a new `MuJoCoDriver`
    explicit MuJoCoDriver()
        : window_(std::make_shared<mjWindowContext>()), sim_(std::make_shared<mjSimContext>()),
          snapshot_(std::make_shared<mjSimSnapshot>())
    {
    }

    ~MuJoCoDriver()
    {
        mjFreeModel(sim_.get());
    }

    /// @brief Get the window context
    mjWindowContext* getWindowContext() const noexcept
    {
        return window_.get();
    }

    /// @brief Get the simulation context
    mjSimContext* getSimContext() const noexcept
    {
        return sim_.get();
    }

    /// @brief Get the simulation snapshot
    mjSimSnapshot* getSnapshot() const noexcept
    {
        return snapshot_.get();
    }

    /// @brief Load a model/scene from a MJ XML file
    bool loadModel(std::filesystem::path model_path, std::string& error)
    {
        error.resize(256);
        bool ok = mjLoadModel(sim_.get(), model_path.c_str(), error.data());
        // error.resize(std::strlen(error.c_str()));
        return ok;
    }

    /// @brief Open the simulation window
    bool openWindow(int width, int height, std::string_view title)
    {
        return mjOpenWindow(sim_.get(), window_.get(), width, height, title.cbegin());
    }

    /// @brief Check if the window should close
    bool shouldCloseWindow()
    {
        return mjShouldWindowClose(window_.get());
    }

    /// @brief Close the window
    void closeWindow()
    {
        mjCloseWindow(window_.get());
    }

    /// @brief Step the simulation
    void step(mjtNum duration, mjtNum sim_dt)
    {
        mjStep(sim_.get(), snapshot_.get(), duration, sim_dt);
    }

    /// @brief Render the window
    void render()
    {
        mjRender(sim_.get(), window_.get());
    }

    /// @brief Reset the sim environment
    void reset()
    {
        mjResetEnvironment(sim_.get());
    }

    /// @brief Set the global MuJoCo context to `this`
    void setActive() const noexcept
    {
        _mjWindowContext() = window_;
        _mjSimContext() = sim_;
    }

    inline auto now() const noexcept
    {
        using namespace std::chrono;
        auto t = duration_cast<steady_clock::duration>(duration<double>(snapshot_->time));
        return steady_clock::time_point{t};
    }

    inline std::span<const mjtNum, 3UL> readBodyPosition() const noexcept
    {
        return std::span{snapshot_->body_position};
    }

    inline std::span<const mjtNum, 4UL> readBodyOrientation() const noexcept
    {
        return std::span{snapshot_->body_quaternion};
    }

    inline std::span<const mjtNum, 3UL> readBodyLinearVelocity() const noexcept
    {
        return std::span{snapshot_->body_linear_velocity};
    }

    inline std::span<const mjtNum, 3UL> readBodyAngularVelocity() const noexcept
    {
        return std::span{snapshot_->body_angular_velocity};
    }

    inline std::span<const mjtNum, 3UL> readBodyLinearAcceleration() const noexcept
    {
        return std::span{snapshot_->body_linear_acceleration};
    }

    inline std::span<const mjtNum, 3UL> readBodyAngularAcceleration() const noexcept
    {
        return std::span{snapshot_->body_angular_acceleration};
    }

    inline std::span<const mjtNum> readJointPositions() const noexcept
    {
        return std::span{snapshot_->joint_positions};
    }

    inline std::span<const mjtNum> readJointVelocities() const noexcept
    {
        return std::span{snapshot_->joint_velocities};
    }

    inline std::span<const mjtNum> readJointAccelerations() const noexcept
    {
        return std::span{snapshot_->joint_accelerations};
    }

    template <std::ranges::range TauRange> inline void setJointTorques(const TauRange& taus) noexcept
    {
        auto& controls = snapshot_->joint_controls;
        controls.resize(std::ranges::size(taus));
        std::ranges::copy(taus, controls.begin());
    }

    template <std::ranges::range TauRange>
    inline void setJointTorques(const TauRange& taus, std::size_t offset, std::size_t count) noexcept
    {
        auto& controls = snapshot_->joint_controls;
        const std::size_t n = std::min<std::size_t>(count, std::ranges::size(taus));
        if (controls.size() < offset + n)
            controls.resize(offset + n, 0.0);
        std::ranges::copy(taus | std::views::take(n), controls.begin() + static_cast<std::ptrdiff_t>(offset));
    }

  private:
  
    std::shared_ptr<mjWindowContext> window_;
    std::shared_ptr<mjSimContext> sim_;
    std::shared_ptr<mjSimSnapshot> snapshot_;
};
} // namespace mj
