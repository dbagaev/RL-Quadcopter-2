import csv
from pathlib import Path


def learn_episode(i_episode, agent, task):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state

        if done:
            print("Episode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                i_episode+1, agent.episode_score, agent.best_score, agent.noise_scale))  # [debug]
            break


def play_episode(i_episode, agent, task, log_stdout=True):
    state = agent.reset_episode()
    eposide_score = 0.0

    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity'] + ['rotor_speed{}'.format(i+1) for i in range(task.num_actions)]

    # Run the simulation, and save the results.
    result_dir = Path.cwd() / 'results'
    if not result_dir.exists():
        result_dir.mkdir()
    result_file = result_dir / '{}-ep-{:04d}.log.csv'.format(task.task_name, i_episode)
    with open(str(result_file), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)

        while True:
            action = agent.act_target(state)
            next_state, reward, done = task.step(action)

            if log_stdout:
                action_str = ', '.join('{:7.3f}'.format(a) for a in action)
                position_str = ', '.join('{:7.3f}'.format(a) for a in state[0:12])
                print('{:5.2f}: [{}] => [{}], R = {:7.3f}'.format(task.sim.time, position_str, action_str, reward))

            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(action)
            writer.writerow(to_write)

            eposide_score += reward
            state = next_state
            if done:
                if log_stdout:
                    print("Act episode {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                        i_episode+1, eposide_score, agent.best_score or 0, agent.noise_scale))  # [debug]
                break


def learn_agent(num_episodes, agent, task, log_episode_every=100, episode_start=0):
    labels = ['episode', 'reward']

    # Run the simulation, and save the results.
    result_dir = Path.cwd() / 'results'
    if not result_dir.exists():
        result_dir.mkdir()
    result_file = result_dir / '{}-rewards.csv'.format(task.task_name)

    result_file_mode = 'w' if episode_start == 0 else 'a'

    with open(str(result_file), result_file_mode) as csvfile:
        writer = csv.writer(csvfile)
        if episode_start == 0:
            writer.writerow(labels)

        for i_episode in range(episode_start, episode_start+num_episodes):
            if i_episode % log_episode_every == 0:
                play_episode(i_episode, agent, task)
            learn_episode(i_episode, agent, task)

            to_write = [i_episode, agent.episode_score]
            writer.writerow(to_write)

        # Make final output
        play_episode(episode_start+num_episodes, agent, task)
