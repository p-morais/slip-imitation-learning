from matplotlib import pyplot as plt
import numpy as np
from torch.autograd import Variable as V
from torch import Tensor

def progress_plot(fig, ax, xlim, data_plt, data):
    data_plt.set_data(np.arange(data.shape[0]), data)
    ax.relim()
    ax.autoscale_view()
    ax.set_xlim(0, xlim)
    plt.pause(0.00001)
    fig.canvas.draw()

def plot_policy(env, action_dim, obs_dim, trj_len, imitation_policy, expert_policy):
        y_plt_exp = np.zeros((trj_len, action_dim))
        y_plt = np.zeros((trj_len, action_dim))
        X_plt = np.zeros((trj_len, obs_dim))

        obs = env.reset()
        for t in range(trj_len):
            X_plt[t, :] = obs
            action = imitation_policy(V(Tensor(obs[None, :]))).data.numpy()
            y_plt_exp[t, :] = expert_policy(V(Tensor(obs[None, :]))).data.numpy()
            y_plt[t, :] = action
            obs = env.step(action)[0]
            env.render()

        plt.subplot(231)
        plt.title("learned action one")
        plt.plot(np.arange(trj_len), y_plt[:, 0])

        plt.subplot(232)
        plt.title("learned action two")
        plt.plot(np.arange(trj_len), y_plt[:, 1])

        plt.subplot(233)
        plt.title("learned action three")
        plt.plot(np.arange(trj_len), y_plt[:, 2])

        plt.subplot(234)
        plt.title("expert action one")
        plt.plot(np.arange(trj_len), y_plt_exp[:, 0])

        plt.subplot(235)
        plt.title("expert action two")
        plt.plot(np.arange(trj_len), y_plt_exp[:, 1])

        plt.subplot(236)
        plt.title("expert action three")
        plt.plot(np.arange(trj_len), y_plt_exp[:, 2])
