import numpy as np
import gym
import gym_minigrid
import pickle
import matplotlib.pyplot as plt
import imageio

MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

def step_cost(action):
    # You should implement the stage cost by yourself
    # Feel free to use it or not
    # ************************************************
    return 0 # the cost of action

def step(env, action):
    '''
    Take Action
    ----------------------------------
    actions:
        0 # Move forward (MF)
        1 # Turn left (TL)
        2 # Turn right (TR)
        3 # Pickup the key (PK)
        4 # Unlock the door (UD)
    '''
    actions = {
        0: env.actions.forward,
        1: env.actions.left,
        2: env.actions.right,
        3: env.actions.pickup,
        4: env.actions.toggle
        }

    _, _, done, _ = env.step(actions[action])
    return step_cost(action), done

def generate_random_env(seed, task):
    ''' 
    Generate a random environment for testing
    -----------------------------------------
    seed:
        A Positive Integer,
        the same seed always produces the same environment
    task:
        'MiniGrid-DoorKey-5x5-v0'
        'MiniGrid-DoorKey-6x6-v0'
        'MiniGrid-DoorKey-8x8-v0'
    '''
    if seed < 0:
        seed = np.random.randint(50)
    env = gym.make(task)
    env.seed(seed)
    env.reset()
    return env

def load_env(path):
    '''
    Load Environments
    ---------------------------------------------
    Returns:
        gym-environment, info
    '''
    with open(path, 'rb') as f:
        env = pickle.load(f)
    
    info = {
        'height': env.height,
        'width': env.width,
        'init_agent_pos': env.agent_pos,
        'init_agent_dir': env.dir_vec
        }
    
    for i in range(env.height):
        for j in range(env.width):
            if isinstance(env.grid.get(j, i),
                          gym_minigrid.minigrid.Key):
                info['key_pos'] = np.array([j, i])
            elif isinstance(env.grid.get(j, i),
                            gym_minigrid.minigrid.Door):
                info['door_pos'] = np.array([j, i])
            elif isinstance(env.grid.get(j, i),
                            gym_minigrid.minigrid.Goal):
                info['goal_pos'] = np.array([j, i])    
            
    return env, info

def save_env(env, path):
    with open(path, 'wb') as f:
        pickle.dump(env, f)

def plot_env(env):
    '''
    Plot current environment
    ----------------------------------
    '''
    img = env.render('rgb_array', tile_size=32)
    plt.figure()
    plt.imshow(img)
    plt.show()
    return img

def draw_gif_from_seq(seq, env, path='./gif/doorkey.gif'):
    '''
    Save gif with a given action sequence
    ----------------------------------------
    seq:
        Action sequence, e.g [0,0,0,0] or [MF, MF, MF, MF]
    
    env:
        The doorkey environment
    '''
    with imageio.get_writer(path, mode='I', duration=0.8) as writer:
        img = env.render('rgb_array', tile_size=32)
        writer.append_data(img)
        for act in seq:
            img = env.render('rgb_array', tile_size=32)
            step(env, act)
            writer.append_data(img)
    print('GIF is written to {}'.format(path))
    return
    
