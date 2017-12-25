#####
#
# this script is more less a copy of the example script in
# https://neurocats.github.io/2017/09/27/PolicyGradient/
#
#####

import gym
import numpy as np
import copy

env = gym.make('MountainCar-v0')



def get_policy(theta, state, gradient=None):
    global env
    # initialisation of variables
    output_policy = []
    phi_values = []
    sum = 0
    theta = theta.reshape(3)

    # actions:
    #       0 -- push left
    #       1 -- no push
    #       2 -- push right

    for action in range(env.action_space.n):
        phi = np.zeros(shape=3)
        p = state[0]    # position
        v = state[1]    # velocity

        # define features
        if v < 0 and action == 0: # velocity is negative (going backwards) and actions is push left
            phi[0] = 1
        if v >= 0 and action == 2: # velocity is negative (going backwards) and actions is push right
            phi[2] = 1
        if p > 0 and action == 2:
            phi[2] = 1 # position of the car


        phi_values.append(phi)

        y = np.exp(np.dot(np.transpose(phi), theta))
        sum += y
        output_policy = np.append(output_policy, y)

    # calculate policy outputs for all actions
    output_policy = ( 1 / sum ) * output_policy
    phi_values = np.array(phi_values)

    if gradient is None:
        return output_policy
    else:
        return output_policy, phi_values


def action_selection(policy):
    x = np.random.uniform(size=1) # random number given by uniform distribution
    # ?? what is this
    if x < policy[0]:
        action = 0
    elif x < (policy[0] + policy[1]):
        action = 1
    else:
        action = 2

    return action

def inference(theta):
    global env
    state = env.reset() # start new episode
    env.render()        #visualize the starting state
    finished = False
    while not finished:
        policy = get_policy(theta, state)
        action = action_selection(policy)
        state, _, finished, _ = env.step(action)
        env.render()

def learn(theta=None):
    if theta is None:
        epsilon = 50
        theta = np.random.uniform(-epsilon, epsilon, 3)
    num_episodes = 50 # number of parameter updates

    for i in range(num_episodes):
        state = env.reset()
        finished = False
        episode = [state]
        T = 0 # counter until terminal state

        # first play one episode with current policy
        while not finished:
            policy = get_policy(theta, state)
            action = action_selection(policy)
            # decide on action write policy representation

            # take action and receive feedback
            state, reward, finished, _ = env.step(action)

            # save for monte carlo estimate
            episode.append(action)
            episode.append(reward)
            episode.append(state)
            T += 1

        # adapt parameters of policy representation
        R = 0
        gamma = 0.99
        alpha = 0.02
        #alpha = 1./((i+1)**2)
        for t in reversed(range(0,T-1,3)):
            index = (t+1)*3
            state = episode[index - 3]
            action = episode[index - 2]
            reward = episode[index - 1]

            R = gamma * R + reward
            policy, phi_values = get_policy(theta, state, True)
            #print(phi_values[action])
            phi_s_a = phi_values[action]
            gradient = copy.deepcopy(phi_values[action])

            for j in  range(env.action_space.n):
                phi_s_c = phi_values[j]
                gradient -= policy[j] * phi_values[j]

            theta += alpha * np.power(gamma, t) * R * gradient
            #print(theta)

    return theta


def run():
    theta = learn()
    starting_theta = theta
    for i in range(1000):
        inference(theta)
        theta = learn(theta)
        print(str(i))
        print(theta)



if __name__ == "__main__":
    run()

