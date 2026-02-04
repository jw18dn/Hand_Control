#pragma once  // Include guard

#include <mujoco/mujoco.h>

enum class Body {
    UNIVERSE,
    PALM,
    INDEX_1,
    INDEX_2,
    INDEX_EE,
    MIDDLE_1,
    MIDDLE_2,
    MIDDLE_EE,
    THUMB_1,
    THUMB_2,
    THUMB_3,
    THUMB_EE,
    COUNT
};

static constexpr std::array<const char*, static_cast<size_t>(Body::COUNT)> BODY_NAMES = {
    "universe",   // UNIVERSE
    "PALM_RZ",    // PALM
    "I_L1",       // INDEX_1
    "I_L2",       // INDEX_2
    "I_L3",       // INDEX_EE
    "P_L1",       // MIDDLE_1
    "P_L2",       // MIDDLE_2
    "P_L3",       // MIDDLE_EE
    "T_L1",       // THUMB_1
    "T_L2",       // THUMB_2
    "T_L3",       // THUMB_3
    "T_L4"        // THUMB_EE
};

inline const char* getBodyName(Body body) {
    return BODY_NAMES[static_cast<size_t>(body)];
}

inline int getBodyRef(const mjModel* m, Body body) {
    return mj_name2id(m, mjOBJ_BODY, getBodyName(body));
}