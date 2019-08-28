'''
Created on Aug 22, 2019

@author: nboutin
'''
import gym
from gym import wrappers, logger


class SimpleAgent():

    def __init__(self, action_space):
        self.action_space = action_space

    def do(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':

    logger.set_level(logger.INFO)

#     env = gym.make('CartPole-v0')
    env = gym.make('Copy-v0')
#     env = gym.make('MountainCar-v0')
#     env = gym.make('MsPacman-v0')

    outdir = 'results'
    env = wrappers.Monitor(env, directory=outdir, force=True)

    print("action:", env.action_space)
    print("observation:", env.observation_space)

    agent = SimpleAgent(env.action_space)

    EPISODE_COUNT = 2
    reward = 0
    done = False

    for i in range(EPISODE_COUNT):
        print("Episode ", i)
        observation = env.reset()
        while True:
            env.render()

            action = agent.do(observation, reward, done)
#             print("action:", action)

            observation, reward, done, info = env.step(action)

            if done:
                break
    env.close()
