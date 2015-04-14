import matplotlib.pyplot as plt
import numpy as np

class Plot(object):
    """
    Wrapper class for collecting all trials to use in visualization methods.

    Parameters
    ----------
    dict_trial: dictionary of lists, where each object in the list is a Trial
        Key is the name of the learner, and value is a list of trials
        for that learner using different parameter settings.
    """
    def __init__(self, dict_trial):
        self.dict_trial = dict_trial

        self.colors = ['r', 'b', 'g', 'm', 'c', 'y']
        self.line_type = ['-', '--', '-.']

    def cum_rewards_by_iteration(self):
        """
        Plot B.
        y-axis: Sum of all rewards.
        x-axis: Iteration of trial(s).
        """
        self.__rewards_by_idx("cum", "iters")

    def rewards_by_episode(self):
        """
        Plot C.
        y-axis: Immediate reward.
        x-axis: Episode of trial(s).
        """
        self.__rewards_by_idx("imm", "epi")

    def cum_rewards_by_prob_start(self):
        """
        Plot F.
        y-axis: Sum of all the rewards.
        x-axis: Pr(return to start).
        """
        self.__rewards_by_prob_start("cum")

    def end_rewards_by_prob_start(self):
        """
        Plot G.
        y-axis: Sum of all the rewards in the last 100 iterations.
        x-axis: Pr(return to start).
        """
        self.__rewards_by_prob_start("end")

    def cum_rewards_by_act_err_prob(self):
        """
        Plot I.
        y-axis: Sum of all the rewards.
        x-axis: Action-error probability.
        """
        self.__rewards_by_act_err_prob("cum")

    def end_rewards_by_act_err_prob(self):
        """
        Plot J.
        y-axis: Sum of all the rewards in the last 100 iterations.
        x-axis: Action-error probability.
        """
        self.__rewards_by_act_err_prob("end")

    def __rewards_by_idx(self, reward_type, idx_type):
        """
        reward_type: "cum" or "imm"
        idx_type: "iters" or "epi"
        """
        i = 0
        for key,value in self.dict_trial.items():
            color = self.colors[i]
            j = 0
            for trial in value:
                line_type = self.line_type[j]
                if reward_type == "cum" and idx_type == "iters":
                    array = trial.array_rewards_by_iteration.cumsum(axis=1)
                elif reward_type == "imm" and idx_type == "epi":
                    array = trial.array_rewards_by_episode
                else:
                    raise Exception("Arguments not specified correctly.")
                x = np.arange(array.shape[1])
                mean = array.mean(axis=0)
                if j == 0:
                    plt.plot(x, mean, color+line_type, label=key)
                else:
                    plt.plot(x, mean, color+line_type)
                j += 1
            i += 1
        if reward_type == "cum" and idx_type == "iters":
            plt.title("Cumulative reward by iteration")
            plt.ylabel("Cumulative reward")
            plt.xlabel("Iteration")
            plt.legend(loc=2)
        elif reward_type == "imm" and idx_type == "epi":
            plt.title("Immediate reward by episode")
            plt.ylabel("Immediate reward")
            plt.xlabel("Episode")
            plt.legend(loc=4)
        plt.show()

    def __rewards_by_prob_start(self, reward_type):
        """
        reward_type: "cum" or "end"
        """
        i = 0
        for key,value in self.dict_trial.items():
            color = self.colors[i]
            x = np.arange(0, 1, 0.1)
            means = np.zeros(len(value))
            j = 0
            for trial in value:
                if reward_type == "cum":
                    array = trial.array_rewards_by_iteration.sum(axis=1)
                elif reward_type == "end":
                    array = trial.array_rewards_by_iteration[:,-100:].sum(axis=1)
                else:
                    raise Exception("Arguments not specified correctly.")
                means[j] = array.mean(axis=0)
                j += 1
            plt.plot(x, means, color, label=key)
            i += 1
        if reward_type == "cum":
            plt.title("Cumulative reward by prob(return_start)")
            plt.ylabel("Cumulative reward")
        elif reward_type == "end":
            plt.title("End reward by prob(return_start)")
            plt.ylabel("End reward (sum of last 100 iterations)")
        plt.xlabel("Prob(return_start)")
        plt.legend()
        plt.show()

    def __rewards_by_act_err_prob(self, reward_type):
        """
        reward_type: "cum" or "end"
        """
        i = 0
        for key,value in self.dict_trial.items():
            color = self.colors[i]
            j = 0
            for trial_list in value:
                line_width = np.linspace(0.5, 3, endpoint=True, num=len(value))[j]
                x = np.arange(0, 0.55, 0.05)
                means = np.zeros(len(trial_list))
                for k,trial in enumerate(trial_list):
                    if reward_type == "cum":
                        array = trial.array_rewards_by_iteration.sum(axis=1)
                    elif reward_type == "end":
                        array = trial.array_rewards_by_iteration[:,-100:].sum(axis=1)
                    else:
                        raise Exception("Arguments not specified correctly.")
                    means[k] = array.mean(axis=0)
                if line_width == 1:
                    plt.plot(x, means, color, linewidth=line_width, label=key)
                else:
                    plt.plot(x, means, color, linewidth=line_width)
                j += 1
            i += 1
        if reward_type == "cum":
            plt.title("Cumulative reward by action-error probability (thicker=larger epsilon)")
            plt.ylabel("Cumulative reward")
        elif reward_type == "end":
            plt.title("End reward by action-error probability (thicker=larger epsilon)")
            plt.ylabel("End reward")
        plt.xlabel("Action-error probability")
        plt.legend()
        plt.show()
