class Evaluator(object):
    def __init__(self, args):
        self.validation_episodes = args.validation_episodes
        self.action_repeat = args.action_repeat
        self.max_episode_length = args.max_episode_length // self.action_repeat

    def __call__(self, env, policy, vis=False):
        result = []
        for _ in range(self.validation_episodes):
            observation = env.reset()
            episode_steps = 0
            episode_rewards = 0.

            done = False
            while not done and episode_steps <= self.max_episode_length:
                action = policy(observation)
                # action repeat
                for _ in range(self.action_repeat):
                    _, reward, done, _ = env.step(action)
                    episode_rewards += reward
                    if vis:
                        env.render()
                    if done:
                        break
                episode_steps += 1
            result.append(episode_rewards)

        return result
