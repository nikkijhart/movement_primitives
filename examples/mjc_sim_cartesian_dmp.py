"""
=================================
Simulate a DMP in Cartesian Space
=================================

A normal DMP is used to represent a Cartesian trajectory given by positions
and quaternions.
"""
print(__doc__)

mjc_dt: float = 0.002

import time
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import mjc_furniture_bench
from mjc_furniture_bench.controllers.diffik_nullspace import DiffIKController

from movement_primitives.dmp import DMP, CartesianDMP

execution_time = 3.0

def get_ee_pos(model, data, ee_idx):
    pos = np.copy(data.site_xpos[ee_idx])
    ee_rot = R.from_matrix(data.site_xmat[ee_idx].reshape((3,3)))
    rot = ee_rot.as_quat(scalar_first=True) # wxyz
    return np.append(pos, rot)

def get_ee_vel(model, data, ee_idx):
    return np.zeros(7)
    return compute_jacobian(model, data, ee_idx) @ data.qvel[:9]

def compute_jacobian(model, data, ee_idx):
    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))

    mujoco.mj_jacSite(model, data, jac_pos, jac_rot, ee_idx)
    jac = np.concatenate((jac_pos[:, :9],
                            jac_rot[:, :9]), axis=0)
    return jac

def main() -> None:

    assert mujoco.__version__ >= "3.3.0", "Please upgrade to mujoco 3.3.0 or later."

    model = mujoco.MjModel.from_xml_path(mjc_furniture_bench.__path__[0] + "/franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)

    model.opt.timestep = mjc_dt

    # Reset the simulation.
    key_name = "home"
    data.qpos[:9] = model.key(key_name).qpos[:9]
    mujoco.mj_step(model, data, nstep=1)

    ee_idx =model.site('EEF').id

    controller = DiffIKController(model, data, "EEF", config=[1.0, 1.0])

    # Set up DMP
    dmp = DMP(n_dims = 7, execution_time=execution_time, dt=mjc_dt,
          n_weights_per_dim=10, int_dt=mjc_dt)
    Y = np.zeros((int(execution_time / mjc_dt), 7))
    T = np.linspace(0, 1, len(Y))
    start = get_ee_pos(model, data, ee_idx)
    sigmoid = 0.5 * (np.tanh(1.5 * np.pi * (T - 0.5)) + 1.0)
    Y[:, [0, 2, 3, 4, 5, 6]] = start[[0, 2, 3, 4, 5, 6]]
    Y[:, 1] = start[1] - 0.2 + 0.4 * sigmoid + np.linspace(-0.4, 0.4, len(Y)) ** 2 + 0.036438676001447734
    dmp.imitate(T, Y)
    Z = np.copy(Y)
    Z[:,1] -= 0.1
    goal = Y[-1]
    dmp.configure(start_y=Z[0], goal_y=goal)

    # set up data to display
    P = []
    dP = []
    V = []
    dV = []

    curr_pos = np.zeros(7)

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        i = 0
        
        while viewer.is_running() and not np.allclose(curr_pos, goal, atol=1e-2):
            step_start = time.time()

            curr_pos = get_ee_pos(model, data, ee_idx)
            curr_vel = get_ee_vel(model, data, ee_idx)
            pos, vel = dmp.step(curr_pos, curr_vel)

            if len(dP) >= 1:
                P.append(curr_pos)
                V.append(curr_vel)

            dP.append(pos)
            dV.append(vel)

            data.ctrl[:7] = controller.compute_control_signal(pos[:3], pos[3:7])[:7]
            data.ctrl[7] = 0

            mujoco.mj_step(model, data, nstep=1)

            viewer.sync()
            time_until_next_step = mjc_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            i+=1

        P = np.asarray(P)
        dP = np.asarray(dP)
        V = np.asarray(V)
        dV = np.asarray(dV)

        print("error between desired and actual: " + str(np.sqrt(np.sum(np.power(dP[-1,:3]-P[-1,:3], 2)))))

        plot_dim = 1
        plt.plot(Y[:, plot_dim], label="Demo")
        plt.scatter([[0, len(Y)]], [[Y[0, plot_dim], Y[-1, plot_dim]]])
        plt.plot(P[:, plot_dim], label="Actual")
        plt.scatter([[0, len(P)]], [[P[0, plot_dim], P[-1, plot_dim]]])
        plt.plot(dP[:, plot_dim], label="Desired")
        plt.scatter([[0, len(dP)]], [[dP[0, plot_dim], dP[-1, plot_dim]]])
        T, Y = dmp.open_loop(run_t=execution_time)
        plt.plot(Y[:, plot_dim], label="Open loop")
        plt.scatter([[0, len(Y)]], [[Y[0, plot_dim], Y[-1, plot_dim]]])
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
