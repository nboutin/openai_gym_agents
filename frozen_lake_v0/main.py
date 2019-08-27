'''
Created on Aug 22, 2019
@author: nboutin

https://github.com/openai/gym/wiki/FrozenLake-v0

The agent controls the movement of a character in a grid world. 
Some tiles of the grid are walkable, and others lead to the agent 
falling into the water. Additionally, the movement direction 
of the agent is uncertain and only partially depends on the 
chosen direction. The agent is rewarded for finding a 
walkable path to a goal tile.

The surface is described using a grid like the following:

    SFFF       (S: starting point, safe)
    FHFH       (F: frozen surface, safe)
    FFFH       (H: hole, fall to your doom)
    HFFG       (G: goal, where the frisbee is located)
    
    The episode ends when you reach the goal or fall in a hole. 
    You receive a reward of 1 if you reach the goal, and zero otherwise.

Environment Id     Observation Space     Action Space     Reward Range     tStepL     Trials     rThresh
FrozenLake-v0     Discrete(16)             Discrete(4)     (0, 1)         100         100        0.78


Solved Requirements:
Reaching the goal without falling into hole over 100 consecutive trials.
'''
import os
import neat
import gym
from math import pow
import numpy as np

NUM_CORES = 1


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = simulate(net, config)


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return simulate(net, config)


env = gym.make('FrozenLake-v0')
env.seed(0)


def simulate(net, config):
    ''' return: fitness'''
    fitness = config.fitness_threshold
    EPISODE_COUNT = int(config.fitness_threshold)

    for i in range(EPISODE_COUNT):
        observation = env.reset()
        while True:
            #                 env.render()
            inputs = [0] * int(env.observation_space.n)
            inputs[observation] = 1
            output = net.activate(inputs)
            action = np.argmax(output)  # return indice of max element
            observation, reward, done, info = env.step(action)

            if done:
                fitness -= pow(reward - 1, 2)
                break
    return fitness


def run(config_file):

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
    if NUM_CORES == 1:
        winner = p.run(eval_genomes)
    else:
        pe = neat.ParallelEvaluator(NUM_CORES, eval_genome)
        winner = p.run(pe.evaluate)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
#     print('\nOutput:')
#     winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
#     for i in inputs:
#         output = winner_net.activate([float(x) for x in i])
#         print("  input {!r}, expected output {!r}, got {!r}".format(i, majorityFunction(*i), output))

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
