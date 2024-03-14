# ECE276B SP20 HW1 Programming Assignment


## Instruction
### 1. doorkey.py
You can run this script to plan an optimal sequence with minimal costs to navigate from the start state of the agent to the goal position, change the env_path in the main() function if you want to try with diffenrent environment.

### 2. utils.py
You might find some useful tools in utils.py
- **step()**: Move your agent
- **generate_random_env()**: Generate a random environment for debugging
- **load_env()**: Load the test environments
- **save_env()**: Save the environment for reproducing results
- **plot_env()**: For a quick visualization of your current env, including: agent, key, door, and the goal
- **draw_gif_from_seq()**: Draw and save a gif image from a given action sequence. **Please notice that you have to submit the gif!**

### 3. example.py
The example.py shows you how to interact with the utilities in utils.py, and also gives you some examples of interacting with gym-minigrid directly.
