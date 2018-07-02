import time
import numpy as np


def train(agent, env, evaluator, writer, args):
    vis = args.vis
    apply_noise = args.apply_noise
    action_repeat = args.action_repeat
    max_episode_length = args.max_episode_length // action_repeat
    checkpoint_num = 0 if args.resume is None else args.resume_num

    log = step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None

    time_stamp = time.time()
    while step <= args.nb_iterations:
        if observation is None:
            observation = env.reset()
            agent.reset(observation)

        # get action by random or actor network
        if step <= args.warmup and args.resume is None:
            action = agent.random_action()
        else:
            action = agent.select_action(observation, apply_noise=apply_noise)

        # action repeat
        repeat_reward = 0.
        for _ in range(action_repeat):
            observation, reward, done, _ = env.step(action)
            repeat_reward += reward
            if vis:
                env.render()
            if done:
                break
        reward = repeat_reward
        agent.observe(reward, observation, done)

        step += 1
        episode_steps += 1
        episode_reward += reward

        if done or (episode_steps >= max_episode_length and max_episode_length):
            if step > args.warmup:
                # checkpoint
                if episode > 0 and episode % args.checkpoint_interval == 0:
                    checkpoint_num += 1
                    print('[save model] #{} in {}'.format(checkpoint_num, args.output))
                    agent.save_model(args.output, checkpoint_num)
                
                # validation
                if episode > 0 and episode % args.validation_interval == 0:
                    validation_reward = evaluator(env, agent.select_action, vis=vis)
                    print('[validation] episode #{}, reward={}'.format(episode, np.mean(validation_reward)))
                    writer.add_scalar('validation/reward', np.mean(validation_reward), episode)

            writer.add_scalar('train/reward', episode_reward, episode)

            # log
            episode_time = time.time() - time_stamp
            time_stamp = time.time()
            print('episode #{}: reward={}, steps={}, time={:.2f}'.format(
                    episode, episode_reward, episode_steps, episode_time
            ))

            for _ in range(args.nb_train_steps):
                log += 1
                Q, critic_loss, critic_output = agent.update_policy()
                writer.add_scalar('train/Q', Q, log)
                writer.add_scalar('train/critic loss', critic_loss, log)
                writer.add_scalar('train/critic output', critic_output, log)

            observation = None
            episode_steps = 0
            episode_reward = 0
            episode += 1
