class Evaluator(object):
    def __init__(self, args):
        self.validation_episodes = args.validation_episodes

    def __call__(self, env, policy, visualize=False):
        result = []
        for episode in range(self.validation_episodes):
            observation = env.reset()
            episode_steps = 0
            episode_rewards = 0.

            done = False
            while not done:
                action = policy(observation)
                observation, reward, done, info = env.step(action)
                if visualize:
                    env.render()
                episode_steps += 1
                episode_rewards += reward
            result.append(episode_rewards)

        return result
