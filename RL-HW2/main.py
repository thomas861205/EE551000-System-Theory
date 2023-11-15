import gym
import pdb
import numpy as np
import argparse
# import plotting
import itertools

# import ours 
from env import CliffWalkingEnv
from algo import q_learning, sarsa
from utils import *

# define algorithm map
ALGO_MAP = {'q_learning': q_learning,
            'sarsa': sarsa}

def render_trajectory(env, Q):
    """
    Description:
        This is the function for you to render the trajectory in Cliff-Walking Environment.
        You should find different trajectory optimized by SARSA and Q-learning!
    """
    state = env.reset()
    for t in itertools.count():
        action = np.argmax(Q[state])
        next_state, reward, done, _ = env.step(action)
        env.render()
        if done:
            break
        state = next_state

if __name__ == '__main__':
    
    # define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-algo", "--algo", default='q_learning', 
                        choices=ALGO_MAP.keys(),
                        help="algorithm to use")
    parser.add_argument("-episode", "--episode", default=500,
                        help="Training episode")
    parser.add_argument("-render", "--render", action='store_true',
                        help="visualize the result of an algorithm")
    parser.add_argument("-runAll", "--runAll", action='store_true', 
                        help="run all algorithms")
    parser.add_argument("-runAlpha", "--runAlpha", action='store_true', 
                        help="run different alpha")
    args = parser.parse_args()

    # initial environment object
    env = CliffWalkingEnv()

    runSingle = not (args.runAll or args.runAlpha)

    if runSingle:
        # start training
        Q, epi_reward, epi_length = ALGO_MAP[args.algo](env, args.episode)
        # NOTE: you can also use your own plotting function
        plot(np.expand_dims(epi_reward, axis=0), [args.algo])
        plot_epi_length(np.expand_dims(epi_length, axis=0), [args.algo])

        # if you want to render the result...
        if args.render:
            render_trajectory(env, Q)

    # Un-comment this part if you finish all algorithm
    
    if args.runAll:
        # run all the algorithms
        label = ['q_learning', 'sarsa']
        _gather_r = np.zeros([len(label), args.episode])
        _gather_l = np.zeros([len(label), args.episode])
        for alg in label:
            idx = label.index(alg)
            Q, epi_reward, epi_length = ALGO_MAP[alg](env, args.episode)
            _gather_r[idx] = epi_reward
            _gather_l[idx] = epi_length
        plot(_gather_r, ['Q-learning', 'SARSA'])
        plot_epi_length(_gather_l, ['Q-learning', 'SARSA'])


    if args.runAlpha:
        # run all the algorithms
        alpha = [0.01, 0.5, 1]
        alg = args.algo
        label = [alg] * len(alpha)
        _gather_r = np.zeros([len(label), args.episode])
        _gather_l = np.zeros([len(label), args.episode])
        for idx in range(len(label)):
            Q, epi_reward, epi_length = ALGO_MAP[label[idx]](env, args.episode, alpha=alpha[idx])
            _gather_r[idx] = epi_reward
            _gather_l[idx] = epi_length
        plot(_gather_r, ['{}, alpha={}'.format(alg, alp) for alp in alpha])
        plot_epi_length(_gather_l, ['Q-learning', 'SARSA'])

