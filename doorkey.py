import numpy as np
import gym
import matplotlib.pyplot as plt
from utils import *
from example import example_use_of_gym_env
from operator import itemgetter

MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

cost_MF = 1 # move forward cost
cost_TL = 1 # turn left cost
cost_TR = 1 # turn right cost
cost_PK = 0 # pick up key cost
cost_UD = 0 # unlock door cost

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

def replace(lst, idx, new_elem):
    new_list = lst.copy()
    new_list[idx] = new_elem
    return new_list

def construct_state_vector(info, env):
    door = env.grid.get(info['door_pos'][0], info['door_pos'][1])
    agent_pos = env.agent_pos
    agent_dir = env.agent_dir
    is_carrying_key = env.carrying is not None
    is_unlocked = not door.is_locked
    '''
    if is_carrying_key:
        is_carrying_key_flag = 1
    else:
        is_carrying_key_flag = 0
    if is_unlocked:
        is_unlocked_flag = 1
    else:
        is_unlocked_flag = 0
    '''
    state_vector = [agent_pos[0], agent_pos[1], agent_dir, is_carrying_key, is_unlocked]
    return state_vector

def rotate_to_direction(env, current_dir, goal_dir, Value, policy, T):
    env.agent_dir = current_dir
    while current_dir != goal_dir:
        if current_dir == 3 and goal_dir == 0:
            step(env,TR)
            policy.append(TR)
            Value += cost_TR
            T -= 1
            current_dir = env.agent_dir
        elif current_dir == 0 and goal_dir == 3:
            step(env,TL)
            Value += cost_TL
            policy.append(TL)
            T -= 1
            current_dir = env.agent_dir
        else:
            while goal_dir > current_dir:
                step(env,TR)
                policy.append(TR)
                Value += cost_TR
                T -= 1
                current_dir = env.agent_dir
            while goal_dir < current_dir:
                step(env,TL)
                policy.append(TL)
                Value += cost_TL
                T -= 1
                current_dir = env.agent_dir
    return env, current_dir, Value, policy, T

def one_step_forward():
    return 0
 
def is_wall(env,pos):
    is_Wall = False
    if env.grid.get(pos[0],pos[1]) is not None:
        if env.grid.get(pos[0],pos[1]).type == 'wall':
            is_Wall = True
    return is_Wall
    
def calc_value_func(env, initial_state, goal_state, planning_horizon):
    env.agent_pos = [initial_state[0], initial_state[1]]
    env.agent_dir = initial_state[2]
    goal_pos = np.array([goal_state[0], goal_state[1]])
    goal_dir = goal_state[2]
    current_state = initial_state
    T = planning_horizon
    # Value function
    Value = 0
    policy = []
    if is_wall(env, goal_pos):
        Value = np.Infinity
    else:
        while current_state != goal_state:
            current_pos = [current_state[0], current_state[1]]
            current_dir = current_state[2]
            while goal_pos[0] > current_pos[0]:
                current_dir = env.agent_dir
                front_pos = env.front_pos
                if current_dir != 0:
                        env, current_dir, Value, policy, T = rotate_to_direction(env, current_dir, 0, Value, policy, T)
                else:
                        if is_wall(env, front_pos):
                            Value = np.Infinity
                            break
                        else:
                            step(env, MF)
                            # T -= 1
                            current_pos[0] += 1
                            Value += cost_MF
                            policy.append(MF)

            while goal_pos[0] < current_pos[0]:
                current_dir = env.agent_dir
                front_pos = env.front_pos
                if current_dir != 2:
                        env, current_dir, Value, policy, T = rotate_to_direction(env, current_dir, 2, Value, policy, T)
                else:
                        if is_wall(env, front_pos):
                            Value = np.Infinity
                            break
                        else:
                            step(env, MF)
                            # T -= 1
                            current_pos[0] -= 1
                            Value += cost_MF
                            policy.append(MF)
                            
            while goal_pos[1] > current_pos[1]:
                current_dir = env.agent_dir
                front_pos = env.front_pos
                if current_dir != 1:
                    env, current_dir, Value, policy, T = rotate_to_direction(env, current_dir, 1, Value, policy, T)
                    # T -= 1
                else:
                        if is_wall(env, front_pos):
                            Value = np.Infinity
                            break
                        else:
                            step(env, MF)
                            # T -= 1
                            current_pos[1] += 1
                            Value += cost_MF
                            policy.append(MF)
            while goal_pos[1] < current_pos[1]:
                current_dir = env.agent_dir
                front_pos = env.front_pos
                if current_dir != 3:
                    env, current_dir, Value, policy, T = rotate_to_direction(env, current_dir, 3, Value, policy, T)
                else:
                        if is_wall(env, front_pos):
                            Value = np.Infinity
                            break
                        else:
                            step(env, MF)
                            # T -= 1
                            current_pos[1] -= 1
                            Value += cost_MF
                            policy.append(MF)
            
            if bool((env.agent_pos == goal_pos).all()) and env.agent_dir != goal_dir:   
                env, current_dir, Value, policy, T = rotate_to_direction(env, current_dir, goal_dir, Value, policy, T)
            current_state = [current_pos[0], current_pos[1], current_dir, initial_state[3], initial_state[4]]
            # if T < 0:
                # Value = np.Infinity
            if Value == np.Infinity:
                break
    steps = len(policy) 
    if steps > planning_horizon:
        Value = np.Infinity 
    # return env, Value, policy, steps #debug
    return Value, policy

def construct_state_space(env, initial_state, obj_pos):
    # obj_pos must be an array
    state_space = []
    goal_obj = env.grid.get(obj_pos[0], obj_pos[1])
    # must_face_obj = False
    wall_inbetween = False
    ss_height_range = list(range(min(initial_state[1], obj_pos[1]), max(initial_state[1], obj_pos[1])+1))
    # check if wall exists between agent and goal
    for h in ss_height_range:
        grid_inbetween = env.grid.get(initial_state[0], h)
        if grid_inbetween is not None:
            if grid_inbetween.type == 'wall':
                wall_inbetween = True
    if wall_inbetween:
        ss_width_range = list(range(min(initial_state[0], obj_pos[0])-1, max(initial_state[0], obj_pos[0])+2))
    else:
        ss_width_range = list(range(min(initial_state[0], obj_pos[0]), max(initial_state[0], obj_pos[0])+1))
    # build the state space
    for i in ss_width_range:
        for j in ss_height_range:
            for k in list(range(0,4)):
                # remove initial state
                if [i,j,k] != initial_state[0:3] and not is_wall(env, (i,j)) and bool(([i,j] != obj_pos).any()):
                        state_space.append([i, j , k, initial_state[3], initial_state[4]])
    return state_space

def find_terminal_state(env, initial_state, state_space, obj_pos):
    terminal_states = []
    # left node, dir right
    left_node = [obj_pos[0] - 1 , obj_pos[1]]
    if not is_wall(env, left_node):
        terminal_states.append([left_node[0], left_node[1], 0, initial_state[3], initial_state[4]])
    # right node, dir left
    right_node = [obj_pos[0] + 1 , obj_pos[1]]
    if not is_wall(env, right_node):
        terminal_states.append([right_node[0], right_node[1], 2, initial_state[3], initial_state[4]])
    # up node, dir down
    up_node = [obj_pos[0] , obj_pos[1] - 1]
    if not is_wall(env, up_node):
        terminal_states.append([up_node[0], up_node[1], 1, initial_state[3], initial_state[4]])
    # down node, dir up
    down_node = [obj_pos[0] , obj_pos[1] + 1]
    if not is_wall(env, down_node):
        terminal_states.append([down_node[0], down_node[1], 3, initial_state[3], initial_state[4]])    
    terminal_states = intersection(terminal_states, state_space)
    return terminal_states

def find_previous_state(state):
    previous_states = []
    # if dir = 1, down
    if state[2] == 1:
        previous_states.append(replace(state, 1, state[1]-1))
        previous_states.append(replace(state, 2, 0))
        previous_states.append(replace(state, 2, 2))
    # if dir = 0, right   
    elif state[2] == 0:
        previous_states.append(replace(state, 0, state[0]-1))
        previous_states.append(replace(state, 2, 3))
        previous_states.append(replace(state, 2, 1))
    # if dir = 3, up
    elif state[2] == 3:
        previous_states.append(replace(state, 1, state[1]+1))
        previous_states.append(replace(state, 2, 2))
        previous_states.append(replace(state, 2, 0))
    # if dir = 2, left
    elif state[2] == 2:
        previous_states.append(replace(state, 0, state[0]+1))
        previous_states.append(replace(state, 2, 1))
        previous_states.append(replace(state, 2, 3))
    return previous_states
    
def Forward_DP(env, initial_state, obj_pos, policy):
    # construct state space
    state_space = construct_state_space(env, initial_state, obj_pos)
    control_space = [MF, TR, TL, PK, UD]
    cost_space = [1,1,1,0,0]
    # total planning horizon limit
    T = len(state_space)
    all_Values = []
    all_Policies = []
    Values_t = []
    Policies_t = []
    # terminal states setup
    terminal_states = find_terminal_state(env, initial_state, state_space, obj_pos)
    terminal_idx = []
    terminal_values = []
    for ts in terminal_states:
        terminal_idx.append(state_space.index(ts))
    reached = False
    # plan time step t = 0
    all_Values.append([np.Infinity] * len(state_space))
    all_Policies.append([[np.Infinity]] * len(state_space))
    # plan time step t = 1
    Values_t = all_Values[0].copy()
    Policies_t = all_Policies[0].copy()
    for s in state_space:
        pv_states = find_previous_state(s)
        if initial_state in pv_states:
            # update values and policies at t = 1
            Values_t[state_space.index(s)] = cost_space[pv_states.index(initial_state)]
            Policies_t[state_space.index(s)] = [control_space[pv_states.index(initial_state)]]
    all_Values.append(Values_t)
    all_Policies.append(Policies_t)
    # plan time step t = 2:T
    for t in list(range(2,T+1)):
        Values_t = all_Values[t-1].copy()
        Policies_t = all_Policies[t-1].copy()
        for j in state_space:
            pv_states = find_previous_state(j)
            Value_temp, policy_temp = [], []
            # calculate value function and policy from start to current_state first
            Value_sj, policy_sj = calc_value_func(env, initial_state, j, t)
            Value_temp.append(Value_sj)
            policy_temp.append(policy_sj)
            for i in pv_states: 
                cost_ij = cost_space[pv_states.index(i)]
                policy_ij = control_space[pv_states.index(i)]
                if i in intersection(find_previous_state(j), state_space): #and Values_t[state_space.index(i)] != np.Infinity:
                    # calculate value and policy thru i to j
                    # only update when value of i state is not Infinity                        
                    Value_sij = Values_t[state_space.index(i)] + cost_ij
                        # previous policy at i
                    pv_policy = Policies_t[state_space.index(i)].copy()
                    if pv_policy == [np.Infinity]:
                        # policy_sij = policy_sj.copy()
                        policy_sij = [policy_ij]
                    else:
                        pv_policy.append(policy_ij)
                        policy_sij = pv_policy.copy()
                    Value_temp.append(Value_sij)
                    policy_temp.append(policy_sij)
                    # update optimal Values and Policies at this plan step
            Values_t[state_space.index(j)] = min(Value_temp)
            Policies_t[state_space.index(j)] = policy_temp[Value_temp.index(min(Value_temp))]
        all_Values.append(Values_t)
        all_Policies.append(Policies_t)
        if Values_t == all_Values[t-1]:
            break
    for idx in terminal_idx:
        terminal_values.append(Values_t[idx])
    if np.Infinity not in terminal_values:
        reached = True
    current_idx = terminal_idx[terminal_values.index(min(terminal_values))]
    current_state = state_space[current_idx]
    optim_policy = policy + Policies_t[current_idx]
    optim_values = []
    for v in all_Values:
        optim_values.append(v[current_idx])
    
    return current_state, reached, optim_policy, optim_values #@ optimal goal_state

def move_agent(env,current_state):
    env.agent_pos = np.array([current_state[0],current_state[1]])
    env.agent_dir = current_state[2]

def doorkey_problem(info, env_name, env):
    '''
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env
        
        doorkey-6x6-direct.env
        doorkey-8x8-direct.env
        
        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env
        
    Feel Free to modify this fuction
    '''
    initial_state = construct_state_vector(info, env)
    optim_policy = []
    # plan to key
    key_pos = info['key_pos']
    
    # pick up key
    front_pos = env.front_pos
    if bool((front_pos == key_pos).all()):
        step(env,PK)
        optim_policy.append(PK)
        current_state = initial_state
        current_state[3] = True
        Values_to_key = [np.Infinity, 0]
    else:
        current_state, reached, optim_policy, Values_to_key = Forward_DP(env, initial_state, key_pos, optim_policy)
        move_agent(env, current_state)
        front_pos = env.front_pos
        if reached and bool((front_pos == key_pos).all()):
            step(env,PK)
            optim_policy.append(PK)
            current_state[3] = True
    # plot and save
    img = plot_env(env)
    plt.imsave(env_name + '_picked_key.png', img) 
    # plan to door
    door_pos = info['door_pos']
    current_state, reached, optim_policy, Values_to_door = Forward_DP(env, current_state, door_pos, optim_policy)
    move_agent(env, current_state)
    # unlock door
    front_pos = env.front_pos
    if reached and bool((front_pos == door_pos).all()):
        step(env,UD)
        optim_policy.append(UD)
        current_state[4] = True
        # step(env,MF)
        # optim_policy.append(MF)
        # current_state[0], curren_state[1] = env.agent_pos[0], env.agent_pos[1]
    # plot and save
    img = plot_env(env)
    plt.imsave(env_name + '_unlocked_door.png', img) 
    # plan to goal
    goal_pos = info['goal_pos']
    current_state, reached, optim_policy, Values_to_goal = Forward_DP(env, current_state, goal_pos, optim_policy)
    move_agent(env, current_state)
    # plot and save
    img = plot_env(env)
    plt.imsave(env_name + '_reaching_goal.png', img) 
    # reach goal
    if reached and bool((front_pos == goal_pos).all()):
        step(env,MF)
        optim_policy.append(MF)
    # output optimal policy, optimal cost, values to key, values to door, values to goal
    return optim_policy, Values_to_key, Values_to_door, Values_to_goal

    
def main():
    env_path = './envs/doorkey-5x5-normal.env'
    env, info = load_env(env_path) # load an environment
    env_name = env_path.split('/')[2].split('.')[0]
    print(env_name + '\n')
    print('<Environment Info>\n')
    print(info) # Map size
                # agent initial position & direction, 
                # key position, door position, goal position
    print('<================>\n')         
    # Visualize the environment
    img = plot_env(env)
    plt.imsave(env_name + '_initial.png', img) 
    # Get the agent position
    agent_pos = env.agent_pos
    
    # Get the agent direction
    agent_dir = env.dir_vec # or env.agent_dir
    
    # Get the cell in front of the agent
    front_cell = env.front_pos # == agent_pos + agent_dir
    
    # Access the cell at coord: (2,3)
    cell = env.grid.get(2, 3) # NoneType, Wall, Key, Goal
    
    # Get the door status
    door = env.grid.get(info['door_pos'][0], info['door_pos'][1])
    key = env.grid.get(info['key_pos'][0], info['key_pos'][1])
    is_open = door.is_open
    is_locked = door.is_locked
    
    seq, Values_to_key, Values_to_door, Values_to_goal = doorkey_problem(info, env_name, env) # find the optimal action sequence
    # reload environment
    env, info = load_env(env_path)
    draw_gif_from_seq(seq, load_env(env_path)[0]) # draw a GIF & save
    for s in seq:
        cost, done = step(env, s)
    if done:
        print("Reached Goal")
    
    # The number of steps so far
    print('Step Count: {}'.format(env.step_count))
    return seq, Values_to_key, Values_to_door, Values_to_goal

if __name__ == '__main__':
    # example_use_of_gym_env()
    optim_policy, Values_to_key, Values_to_door, Values_to_goal = main()
        
        
    
