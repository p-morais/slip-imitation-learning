import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable as V
from torch import Tensor

from policy import MLP

from util.test import render_policy
from util.data import SplitDataset
from util.logging import progress
from util.plotting import progress_plot
from util.plotting import plot_policy

from controller import controller
from controller import clamp

from matplotlib import pyplot as plt
from slip_env import SlipEnv

from torch.optim.lr_scheduler import LambdaLR

import numpy as np
import torch
import torch.nn as nn
import gym

import time

plt.style.use('ggplot')
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

ENV_NAME = 'Hopper-v1'

TIME_STEP = 0.0002

BATCH_SIZE = 100
VALIDATION_SPLIT = 0.0
N = 128  # default = 128
EPOCHS = 10  # default = 20
TRJ_LEN = 10000  # int(3 / TIME_STEP)  # default = 500
DAGGER_ITR = 100  # default = 20

EVAL_LEN = 50000  # default = 100
DROP_OLD = False  # discard old training data for time efficiency

NAIVE = False

if NAIVE:
    DAGGER_ITR = 1
    EPOCHS *= 10
    TRJ_LEN *= 4


def main():
    env = SlipEnv(0.001)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    X, y = Tensor(TRJ_LEN, obs_dim), Tensor(TRJ_LEN, action_dim)
    epoch_losses = np.zeros(EPOCHS * DAGGER_ITR)

    imitation_policy = MLP(obs_dim, action_dim, (N,), F.tanh)

    #scheduler = LambdaLR(imitation_policy.optimizer,
    #                     lambda itr: 1 / (itr + 1))

    loss_plt, = ax.plot([], [])
    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch")
    plt.tight_layout()

    dagger_init = True
    for d in range(DAGGER_ITR):
        #scheduler.step()

        X_new, y_new = Tensor(TRJ_LEN, obs_dim), Tensor(TRJ_LEN, action_dim)

        obs = env.reset().astype(float)
        for t in range(TRJ_LEN):
            action = imitation_policy(V(Tensor(obs[None, :]))).data.numpy()
            action = action[0].astype(float)

            expert_action = controller(env.cstate, 0, 1, 0)
            if dagger_init:
                obs = env.step(expert_action)[0].astype(float)
                X[t, :], y[t, :] = Tensor(obs), Tensor(expert_action)

            else:
                obs, _, done, _ = env.step(action)
                if done:
                    print(t)
                    X_new, y_new = X_new[0:t, :], y_new[0:t, :]
                    break

                obs = obs.astype(float)
                X_new[t, :], y_new[t, :] = Tensor(obs), Tensor(expert_action)

            # print(expert_action-action)
            env.render()

        if not dagger_init:
            X, y = torch.cat((X, X_new), 0), torch.cat((y, y_new), 0)

        dataset = SplitDataset(X.numpy(), y.numpy())
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for e in range(EPOCHS):
            running_loss = 0
            for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                X_batch, y_batch = V(Tensor(X_batch)), V(Tensor(y_batch))

                running_loss += imitation_policy.fit(X_batch, y_batch)[0]
                progress(e * len(dataloader) + batch_idx,
                         len(dataloader) * EPOCHS,
                         "DAgger iteration: %s / %s" % (d + 1, DAGGER_ITR))

            # plotting stuff
            epoch_losses[d * EPOCHS + e] = running_loss / len(dataloader)
            progress_plot(fig, ax, d * EPOCHS + e, loss_plt, epoch_losses)

        torch.save(imitation_policy, "model001.pkl")
        dagger_init = False

    plt.ioff()
    plt.show()

    """
    env.reset()
    for t in range(5000):
        action = controller(env.cstate, 0, 1, 0)
        env.step(action)
        time.sleep(1/float(1000))
        env.render()
    """

    render_policy(env, imitation_policy, EVAL_LEN)
    plot_policy(env, EVAL_LEN, 1, imitation_policy)
    plt.tight_layout()
    plt.show()

    #plot_policy(env, action_dim, obs_dim, EVAL_LEN, imitation_policy, expert_policy)


def plot_policy(env, trj_len, fig=1, policy=None, obs_dim=11, action_dim=2):
    y = np.zeros((trj_len, action_dim))
    y_exp = np.zeros((trj_len, action_dim))
    X = Tensor(trj_len, obs_dim)

    plt.figure(fig)
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

            env.step(y_exp[t, :])

        print(nn.MSELoss()(V(Tensor(y)), V(Tensor(y_exp))))

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
