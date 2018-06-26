class Evaluator(object):
    def __init__(self, args):
        self.validate_interval = args.validate_interval

    def __call__(self, env, policy, visualize=False):
        result = []
        for _ in range(self.validate_interval):
            observation = env.reset()
            episode_steps = 0
            episode_rewards = 0.

            done = False
            while not done:
                action = policy(observation)
                observation, reward, done, _ = env.step(action)
                if visualize:
                    env.render()
                episode_steps += 1
                episode_rewards += reward
            result.append(episode_rewards)

        return result
