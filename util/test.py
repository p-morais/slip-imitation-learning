from torch.autograd import Variable as V
from torch import Tensor
import time

def render_policy(env, policy, trj_len):
    obs = env.reset()
    for t in range(trj_len):
        action = policy(V(Tensor(obs[None, :]))).data.numpy()[0].astype(float)
        obs, _, done, _ = env.step(action)
        #if done:
        #    break
        env.render()
        time.sleep(0.001)
