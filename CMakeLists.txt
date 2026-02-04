cmake_minimum_required(VERSION 3.10)
project(MuJoCoHandSim)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)  # optional, disables GNU extensions

# --- External Packages ---
find_package(glfw3 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)  # header-only

# --- Specify MuJoCo include and library manually ---
set(MUJOCO_INCLUDE_DIR "$ENV{HOME}/.MuJoCo/build/include")
set(MUJOCO_LIBRARY "$ENV{HOME}/.MuJoCo/build/lib/libmujoco.so")

# --- Create a header-only library ---
add_library(handlib INTERFACE)

# Public include directories
target_include_directories(handlib INTERFACE
    include                 # your project headers
    ${MUJOCO_INCLUDE_DIR}    # MuJoCo headers
    ${EIGEN3_INCLUDE_DIR}    # Eigen headers
)

# Link dependencies used inside headers
target_link_libraries(handlib INTERFACE
    ${MUJOCO_LIBRARY}      # MuJoCo library
    glfw
    Eigen3::Eigen          # Eigen (header-only)
)

# --- Build the main executable ---
add_executable(hand_sim 
    src/main.cpp 
    src/kinematics.cpp 
    src/parse_data.cpp 
    src/states.cpp 
    src/state_logger.cpp 
    src/operational_space_control.cpp
    src/attitude_PD_control.cpp
    src/model_predictive_control.cpp
)

# Make sure C++20 applies to this target
target_compile_features(hand_sim PUBLIC cxx_std_20)

# Link against the header-only library
target_link_libraries(hand_sim PRIVATE handlib)