# vpg_acme_jax
An implementation of vanilla policy gradient in JAX in DeepMind's [Acme](https://github.com/deepmind/acme/tree/master) reinforcement learning framework. Modified from [PPO](https://github.com/deepmind/acme/tree/master/acme/agents/jax/ppo) from the same.
This is mainly intended to be educational, not functional. You should really be using PPO or similar. :)
Speaking of educational, check out the blog posts!

# blog posts
This is part of a [blog post](https://kjabon.github.io/blog/2023/VPGJAX/) explaining the implementation of VPG.
That post follows [two](https://kjabon.github.io/blog/2023/VPG/) [more](https://kjabon.github.io/blog/2023/VPG2/) which explain the theory behind VPG.
Those posts follow [another](https://kjabon.github.io/blog/2023/RL/) on the reinforcement learning (RL) problem.

# requirements
- jax ([install](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier) with gpu support)
- dm-acme
- GPUtil
