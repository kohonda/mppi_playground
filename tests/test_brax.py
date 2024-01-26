"""
Because we want to use GPU accerated simulator for MPPI, I tried to use brax simulator.

# jax cuda install
pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# brax install
pip3 install brax

# How to run
# https://tech.yellowback.net/posts/jax-oom
XLA_PYTHON_CLIENT_MEM_FRACTION=.8 python3 tests/test_brax.py 
"""

from brax.io import image
from brax import envs

import jax

rng = jax.random.PRNGKey(0)
ant = envs.create("ant")

rng, rng_use = jax.random.split(rng)
state = ant.reset(rng_use)

# Too slow, not sure why
qps = [state.pipeline_state]
for _ in range(20):
    rng, rng_use = jax.random.split(rng)
    state = ant.step(state, jax.random.uniform(rng_use, (ant.action_size,)))
    qps.append(state.pipeline_state)

# https://github.com/google/brax/issues/47
# How can i get the rendered image without notebook?
image.render(sys=ant.sys, states=qps, width=320, height=240)
