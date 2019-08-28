'''
Created on Aug 22, 2019
@author: nboutin

Copy-v0

This task involves copying the symbols from the input tape to the output tape. 
Although simple, the model still has to learn the correspondence between input
 and output symbols, as well as executing the move right action on the input tape.

Environment Id     Observation Space     Action Space                             Reward Range     tStepL     Trials     rThresh
Copy-v0             Discrete(6)     Tuple(Discrete(2),Discrete(2), Discrete(5))     (-inf, inf)     200         100     25.0


'''
import os
import neat
import gym
from gym import wrappers
import numpy as np
import multiprocessing as mp

import visualize

CORE_COUNT = mp.cpu_count()
EPISODE_COUNT = 100


def evaluate(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def eval_genome(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)
    return simulate(net, config)


def simulate(net, config):
    ''' return: fitness

        Actions consist of 3 sub-actions:
        - Direction to move the read head (left or right, plus up and down for 2-d envs)
        - Whether to write to the output tape
        - Which character to write (ignored if the above sub-action is 0)

        Reward schedule:
            write a correct character: +1
            write a wrong character: -.5
            run out the clock: -1
            otherwise: 0
    '''
    env = gym.make('Copy-v0')
    fitness = 0

    for i in range(EPISODE_COUNT):
        observation = env.reset()
        net.reset()
        while True:
            inputs = [0] * int(env.observation_space.n)
            inputs[observation] = 1
            output = net.activate(inputs)

            d = output[0:2]
            w = output[2:4]
            c = output[4:]
            action = (np.argmax(d), np.argmax(w), np.argmax(c))
            observation, reward, done, info = env.step(action)

            if done:
                fitness += reward
                break
    env.close()
    return fitness


def run(config_file):

    print("Using ", CORE_COUNT, " core")

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
#     p.add_reporter(neat.Checkpointer(5))

    # Run until a solution is found.
    winner = None
    if CORE_COUNT == 1:
        winner = p.run(evaluate)
    else:
        pe = neat.ParallelEvaluator(CORE_COUNT, eval_genome)
        winner = p.run(pe.evaluate)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    visualize.plot_stats(stats, ylog=False, view=False, filename="fitness.svg")

    name = 'winner'
    g = winner
    visualize.draw_net(config, g, view=False, filename=name + "-net.gv")

    # Execute Winner
    net = neat.nn.RecurrentNetwork.create(winner, config)
    env = gym.make('Copy-v0')
    env = wrappers.Monitor(env, directory="monitor", force=True)
#     env = wrappers.Monitor(env, directory="monitor", video_callable=lambda eid: True, force=True)

    for i in range(EPISODE_COUNT):
        observation = env.reset()
        net.reset()
        while True:
            env.render()
            inputs = [0] * int(env.observation_space.n)
            inputs[observation] = 1
            output = net.activate(inputs)

            d = output[0:2]
            w = output[2:4]
            c = output[4:]
            action = (np.argmax(d), np.argmax(w), np.argmax(c))
            observation, reward, done, info = env.step(action)

            if done:
                break
    env.close()


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)

# class SimpleAgent():
#
#     def __init__(self, action_space):
#         self.action_space = action_space
#
#     def do(self, observation, reward, done):
#         print("obs:", observation)
#         print("reward:", reward)
#         return self.action_space.sample()
#
#
# if __name__ == '__main__':
#
#     logger.set_level(logger.INFO)
#
#     env = gym.make('FrozenLake-v0')
#
#     outdir = 'results'
#     env = wrappers.Monitor(env, directory=outdir, force=True)
#
#     print("action:", env.action_space)
#     print("observation:", env.observation_space)
# #     action: Discrete(4) (0:left, 1:down, 2:right, 3:up)
# #     observation: Discrete(16) For 4x4 square, counting each position from left to right, top to bottom
#
#     agent = SimpleAgent(env.action_space)
#
#     EPISODE_COUNT = 1
#     reward = 0
#     done = False
#
#     for i in range(EPISODE_COUNT):
#         observation = env.reset()
#         while True:
#             env.render()
#
#             action = agent.do(observation, reward, done)
# #             print("action:", action)
#
#             observation, reward, done, info = env.step(action)
# #             print("info:", info)
#
#             if done:
#                 break
#     env.close()
