#include "states.hpp"
#include <mujoco/mujoco.h>

void States::initializeStorage(const mjModel* m)
{
    if (initialized_) return;

    q_meas_W_.resize(m->nq);
    dq_meas_W_.resize(m->nv);
    body_quat_.resize(m->nbody);
    body_iso_.resize(m->nbody);
    initialized_ = true;
}

void States::updateStates(const mjModel* m, const mjData* d) {
    // Update positions (size nq)
    for (int i = 0; i < m->nq; i++) {
        q_meas_W_[i] = d->qpos[i];
    }

    // Update velocities (size nv) - separate loop!
    for (int i = 0; i < m->nv; i++) {
        dq_meas_W_[i] = d->qvel[i];
    }

    for (int i = 0; i < m->nbody; i++)
    {
        Eigen::Quaterniond body_rot(
            d->xquat[4*i + 0],
            d->xquat[4*i + 1],
            d->xquat[4*i + 2],
            d->xquat[4*i + 3]
        );
        body_rot.normalize();

        Eigen::Vector3d body_pos(
            d->xpos[3*i + 0],
            d->xpos[3*i + 1],
            d->xpos[3*i + 2]
        );

        body_iso_[i].setIdentity();
        body_iso_[i].linear()      = body_rot.toRotationMatrix();
        body_iso_[i].translation() = body_pos;
        body_quat_[i] = body_rot;
    }
}

const Eigen::Vector3d States::getBodyTransVel_meas_W(const mjModel* m, const mjData* d, const Body body)
{
    mjtNum vel[6];
    int body_id = getBodyRef(m, body);
    mj_objectVelocity(m, d, mjOBJ_BODY, body_id, vel, 0);
    Eigen::Vector3d vel_return(vel[3], vel[4], vel[5]);

    return vel_return;
}

const Eigen::Vector3d States::getBodyAngVel_meas_W(const mjModel* m, const mjData* d, const Body body)
{
    mjtNum vel[6];
    int body_id = getBodyRef(m, body);
    mj_objectVelocity(m, d, mjOBJ_BODY, body_id, vel, 0);
    Eigen::Vector3d vel_return(vel[3], vel[4], vel[5]);

    return vel_return;
}

const Eigen::Vector3d States::getBodyTransVel_meas_B(const mjModel* m, const mjData* d, const Body body)
{
    mjtNum vel[6];
    int body_id = getBodyRef(m, body);
    mj_objectVelocity(m, d, mjOBJ_BODY, body_id, vel, 1);
    Eigen::Vector3d vel_return(vel[3], vel[4], vel[5]);

    return vel_return;
}

const Eigen::Vector3d States::getBodyAngVel_meas_B(const mjModel* m, const mjData* d, const Body body)
{
    mjtNum vel[6];
    int body_id = getBodyRef(m, body);
    mj_objectVelocity(m, d, mjOBJ_BODY, body_id, vel, 1);
    Eigen::Vector3d vel_return(vel[3], vel[4], vel[5]);

    return vel_return;
}