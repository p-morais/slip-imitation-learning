import torch
from util.test import render_policy
from slip_env import SlipEnv
from controller import controller
import time
from torch.autograd import Variable as V
from matplotlib import pyplot as plt
from torch import Tensor
import numpy as np

plt.style.use('ggplot')

def main():
    print("working")
    env = SlipEnv(0.001)
    model = torch.load("model.pkl")

    plot_policy(env, 2000, model)
    plt.tight_layout();
    plt.show()

    env.reset()
    for t in range(1000):
        action = controller(env.cstate, 0, 1, 0)
        env.step(action)
        time.sleep(0.001)
        env.render()

    render_policy(env, model, 10000)

    plot_policy(env, 1000)

def plot_policy(env, trj_len, policy=None, obs_dim=11, action_dim=2):
    y = np.zeros((trj_len, action_dim))
    y_exp = np.zeros((trj_len, action_dim))
    X = Tensor(trj_len, obs_dim)

    plt.figure(1)
    env.reset()
    steps = []
    if policy is not None:
        for t in range(trj_len):
            X[t, :] = Tensor(env.trunc_state)
            action = policy.forward(V(X[None, t, :])).data

            y[t, :] = action.numpy()[0].astype(float)
            y_exp[t, :] = controller(env.cstate, 0, 1, 0)

            if env.last_trunc_state[-1] != env.trunc_state[-1]:
                steps += [t]

            env.step(y[t, :])

        plt.subplot(221)
        plt.title("learned l_torque")
        plt.plot(np.arange(trj_len), y[:, 0])

        plt.subplot(222)
        plt.title("learned theta_torque")
        plt.plot(np.arange(trj_len), y[:, 1])

        plt.subplot(223)
        plt.title("expert l_torque")
        plt.plot(np.arange(trj_len), y_exp[:, 0])

        plt.subplot(224)
        plt.title("expert theta_torque")
        plt.plot(np.arange(trj_len), y_exp[:, 1])

        #for
        #    plt.axvline(t, color='C1', ls='--')

    else:
        for t in range(trj_len):
            y[t, :] = controller(env.cstate, 0, 1, 0)
            env.step(y[t, :])

        plt.subplot(211)
        plt.plot(np.arange(trj_len), y[:, 0])

        plt.subplot(212)
        plt.plot(np.arange(trj_len), y[:, 1])

if __name__ == "__main__":
    main()
