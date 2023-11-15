import sys
import gym
import itertools
import numpy as np
from collections import defaultdict

def epsilon_greedy_policy(Q, epsilon, num_of_action):
    """
    Description:
        Epsilon-greedy policy based on a given Q-function and epsilon.
        Don't need to modify this :) 
    """
    def policy_fn(obs):
        A = np.ones(num_of_action, dtype=float) * epsilon / num_of_action
        best_action = np.argmax(Q[obs])
        A[best_action] += (1.0 - epsilon)
        return A
    
    return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control.

    Inputs:
        env: Environment object.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action.

    Returns:
        Q: the optimal action-value function, a dictionary mapping state -> action values.
        episode_rewards: reward array for every episode
        episode_lengths: how many time steps be taken in each episode
    """

    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)

    # The policy we're following
    policy = epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    # start training
    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        # print("Episodes({}/{})".format(i_episode, num_episodes))
        state = env.reset()

        for t in itertools.count():
            # raise NotImplementedError('Q-learning NOT IMPLEMENTED')
            pmf = policy(state)
            action = np.random.choice(range(len(pmf)), p=pmf)
            # action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            target = reward + discount_factor * max(Q[next_state])
            Q[state][action] = Q[state][action] + alpha * (target - Q[state][action])
            episode_rewards[i_episode] += reward
            # print(episode_rewards[i_episode])
            # env.render()
            if done:
                break
            state = next_state

        episode_lengths[i_episode] = t

    return Q, episode_rewards, episode_lengths

def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control.

    Inputs:
        env: environment object.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        episode_rewards: reward array for every episode
        episode_lengths: how many time steps be taken in each episode
    """
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)

    # The policy we're following
    policy = epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        # print("Episodes({}/{})".format(i_episode, num_episodes))
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        for t in itertools.count():
            # raise NotImplementedError('SARSA NOT IMPLEMENTED')
            next_state, reward, done, _ = env.step(action)
            episode_rewards[i_episode] += reward
            # print(episode_rewards[i_episode])
            if done:
                Q[state][action] = Q[state][action] + alpha * (reward - Q[state][action])
                break

            action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            target = reward + discount_factor * Q[next_state][next_action]
            Q[state][action] = Q[state][action] + alpha * (target - Q[state][action])

            action = next_action
            state = next_state
        episode_lengths[i_episode] = t

    return Q, episode_rewards, episode_lengths
