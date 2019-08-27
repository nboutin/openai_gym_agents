'''
Created on Aug 22, 2019
'''
import gym

if __name__ == '__main__':
    #     env = gym.make('CartPole-v0')
    env = gym.make('Copy-v0')
#     env = gym.make('MountainCar-v0')
#     env = gym.make('MsPacman-v0')

    print("action:", env.action_space)
    print("observation:", env.observation_space)

    for i_episode in range(2):
        observation = env.reset()
        for t in range(100):
            env.render()
            print("observation:", observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
