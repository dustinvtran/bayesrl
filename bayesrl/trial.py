import numpy as np

class Trial(object):
    """
    Class for running trial(s) for a given agent and task.

    Parameters
    ----------
    agent: Agent
    task: Task
    MIN_ITERATIONS: int
        The minimum number of iterations for a trial.
    MIN_EPISODES: int
        The minimum number of episodes for a trial.
    MAX_EPISODE_ITERATION: int
        The maximum number of iterations for each episode.
    """
    def __init__(self, agent, task, MIN_ITERATIONS=5000, MIN_EPISODES=100, MAX_EPISODE_ITERATION=1000):
        self.agent = agent
        self.task = task
        self.MIN_ITERATIONS = MIN_ITERATIONS
        self.MIN_EPISODES = MIN_EPISODES
        self.MAX_EPISODE_ITERATION = MAX_EPISODE_ITERATION

        self.array_rewards_by_episode = None
        self.array_rewards_by_iteration = None

    def run(self):
        iteration = episode = 0
        rewards_by_iteration = np.zeros(self.MIN_ITERATIONS)
        rewards_by_episode = np.zeros(self.MIN_EPISODES)
        self.agent.reset()

        while iteration < self.MIN_ITERATIONS or episode < self.MIN_EPISODES:
            print "Episode:",episode
            # Initialize the episode.
            self.task.reset()
            if self.task.pomdp:
                self.agent.reset_belief()
            state = self.task.observe()
            reward = None
            cumulative_reward = 0
            episode_iteration = 0

            while episode_iteration < self.MAX_EPISODE_ITERATION:
                # Tell the agent what happened and ask for a next action.
                action = self.agent.interact(reward, state, self.task.is_terminal(state), iteration)

                if self.task.is_terminal(state):
                    # End of episode (happens after interaction so agent can learn from final reward).
                    break

                # Take action A, observe R, S'.
                state, reward = self.task.perform_action(action)

                # Log rewards.
                if iteration < self.MIN_ITERATIONS:
                    rewards_by_iteration[iteration] = reward
                cumulative_reward += reward

                iteration += 1
                episode_iteration += 1

            if episode < self.MIN_EPISODES:
                rewards_by_episode[episode] = cumulative_reward
            episode += 1

        return rewards_by_iteration, rewards_by_episode

    def run_multiple(self, num_trials):
        self.array_rewards_by_episode = np.zeros((num_trials, self.MIN_EPISODES))
        self.array_rewards_by_iteration = np.zeros((num_trials, self.MIN_ITERATIONS))
        for i in xrange(num_trials):
            self.array_rewards_by_iteration[i], self.array_rewards_by_episode[i] = self.run()
