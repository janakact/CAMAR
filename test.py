import jax
from camar import camar_v0

# Initialize random keys
key = jax.random.key(0)
key, key_r, key_a, key_s = jax.random.split(key, 4)

# Create environment (default: random_grid map with holonomic dynamics)
env = camar_v0()
reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)

# Reset the environment
obs, state = reset_fn(key_r)
print(obs.shape)
print(state.physical_state.agent_pos.shape)

# Sample random actions
actions = env.action_spaces.sample(key_a)

# Step the environment
obs, state, reward, done, info = step_fn(key_s, state, actions)