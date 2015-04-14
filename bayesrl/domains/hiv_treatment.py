"""HIV Treatment domain

based on https://bitbucket.org/rlpy/rlpy/src/master/rlpy/Domains/HIVTreatment.py
"""
import numpy as np
from scipy.integrate import odeint
from ..utils import check_random_state

# Original attribution information:
__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann"


def random_policy(state, rng):
    """A policy that picks one of the 4 actions randomly."""
    return rng.choice(4)


def always_do(action):
    """Returns a policy that always takes the given action."""
    def policy(state, rng):
        return action
    return policy


class HIVTreatment(object):
    """
    Simulation of HIV Treatment. The aim is to find an optimal drug schedule.

    **STATE:** The state contains concentrations of 6 different cells:

    * T1: non-infected CD4+ T-lymphocytes [cells / ml]
    * T1*:    infected CD4+ T-lymphocytes [cells / ml]
    * T2: non-infected macrophages [cells / ml]
    * T2*:    infected macrophages [cells / ml]
    * V: number of free HI viruses [copies / ml]
    * E: number of cytotoxic T-lymphocytes [cells / ml]

    **ACTIONS:** The therapy consists of 2 drugs
    (reverse transcriptase inhibitor [RTI] and protease inhibitor [PI]) which
    are activated or not. The action space contains therefore of 4 actions:

    * *0*: none active
    * *1*: RTI active
    * *2*: PI active
    * *3*: RTI and PI active

    **REFERENCE:**

    .. seealso::
        Ernst, D., Stan, G., Gonc, J. & Wehenkel, L.
        Clinical data based optimal STI strategies for HIV:
        A reinforcement learning approach
        In Proceedings of the 45th IEEE Conference on Decision and Control (2006).


    """
    state_names = ("T1", "T2", "T1*", "T2*", "V", "E")
    eps_values_for_actions = np.array([[0., 0.], [.7, 0.], [0., .3], [.7, .3]])
    num_actions = 4

    def __init__(self, logspace=True, dt=5, model_derivatives=None):
        self.logspace = logspace
        if logspace:
            self.statespace_limits = np.array([[-5, 8]] * 6)
        else:
            self.statespace_limits = np.array([[0., 1e8]] * 6)
        if model_derivatives is None:
            model_derivatives = dsdt
        self.model_derivatives = model_derivatives
        self.dt = dt
        self.reset()

    def reset(self):
        self.t = 0
        # non-healthy stable state of the system
        self.state = np.array([163573., 5., 11945., 46., 63919., 24.])

    def observe(self):
        if self.logspace:
            return np.log10(self.state)
        else:
            return self.state

    def perform_action(self, action):
        self.t += 1
        eps1, eps2 = self.eps_values_for_actions[action]
        self.state = odeint(self.model_derivatives, self.state, [0, self.dt],
                            args=(eps1, eps2), mxstep=1000)[-1]
        T1, T2, T1s, T2s, V, E = self.state
        # the reward function penalizes treatment because of side-effects
        reward = - 0.1 * V - 2e4 * eps1 ** 2 - 2e3 * eps2 ** 2 + 1e3 * E

        return reward, self.observe()

    @classmethod
    def generate_batch(cls, num_patients, policy=random_policy, rng=None, episode_length=200, **kw):
        """
        Generate a batch of simulation traces for patients, following a given policy.

        Parameters
        ----------

        num_patients : int
            number of patients to simulate

        policy : function (state, rng) -> action; default: random_policy
            The policy to evaluate, specified as a function of the current state.
            The policy function also takes a random number generator object
            (http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html)

        rng : None, int, or RandomState object; default: None
            For repeatable experiments, you can pass a random state here. See
            http://scikit-learn.org/stable/modules/generated/sklearn.utils.check_random_state.html.

        episode_length : int, default 200
            The length of the simulation. With the default time step of 5 days, the default episode
            length gives 1000-day treatments. Since both the initial and final observations are
            included, the output vectors include `N = episode_length + 1` elements.


        Any other keyword parameters will be passed on to the simulator, including: `dt` (the
        observation/action interval, in days), `logspace` (boolean, whether observations should be
        log-transformed), and `model_derivatives` (to override the simulation dynamics).

        Returns
        -------

        A 3-tuple (state_histories, action_histories, reward_histories):

        state_histories : NumPy ndarray with shape (num_patients, N, 6), `float` dtype
            For each patient, the state vector at each step.

        action_histories : NumPy ndarray with shape (num_patients, N), `int` dtype
            For each patient, the action chosen by the policy at each step.

        reward_histories : NumPy ndarray with shape (num_patients, N), `float` dtype
            For each patient, the reward obtained after taking the corresponding action at
            each step.

        The final action and reward represent moving past the observation window, so
        you may want to ignore them.
        """
        rng = check_random_state(rng)
        state_histories = np.empty((num_patients, episode_length + 1, len(cls.state_names)))
        action_histories = np.empty((num_patients, episode_length + 1), dtype=np.int8)
        reward_histories = np.empty((num_patients, episode_length + 1))

        simulator = cls(**kw)
        for patient in range(num_patients):
            simulator.reset()
            for episode in range(episode_length + 1):
                state_histories[patient, episode] = state = simulator.observe()
                action_histories[patient, episode] = action = policy(state, rng)
                reward, state = simulator.perform_action(action)
                reward_histories[patient, episode] = reward

        return state_histories, action_histories, reward_histories


def visualize_hiv_history(state_history, action_history, handles=None):
    """
    Shows a graph of each concentration and the action taken.

    Returns a "handles" array; if you pass it back in again, the drawing will be faster.
    """

    import matplotlib.pyplot as plt
    history = np.concatenate([state_history.T, action_history[None, :]], axis=0)
    num_dims, num_steps = history.shape
    names = list(HIVTreatment.state_names) + ["Action"]
    colors = ["b", "b", "b", "b", "r", "g", "k"]
    if handles is None:
        handles = []
        fig, axes = plt.subplots(
            num_dims, sharex=True, num="Domain", figsize=(12, 10))
        fig.subplots_adjust(hspace=0.1)
        for i, ax in enumerate(axes):
            d = np.arange(num_steps) * 5
            ax.set_ylabel(names[i])
            ax.locator_params(tight=True, nbins=4)
            handles.append(
                ax.plot(d,
                        history[i],
                        color=colors[i])[0])
        ax.set_xlabel("Days")
    for i in range(num_dims):
        handles[i].set_ydata(history[i])
        ax = handles[i].get_axes()
        ax.relim()
        ax.autoscale_view()
    plt.draw()
    return handles


def dsdt(s, t, eps1, eps2):
    derivs = np.empty_like(s)
    dsdt_(derivs, s, t, eps1, eps2)
    return derivs


def dsdt_(out, s, t, eps1, eps2):
    """
    system derivate per time. The unit of time are days.
    """
    # model parameter constants
    lambda1 = 1e4
    lambda2 = 31.98
    d1 = 0.01
    d2 = 0.01
    f = .34
    k1 = 8e-7
    k2 = 1e-4
    delta = .7
    m1 = 1e-5
    m2 = 1e-5
    NT = 100.
    c = 13.
    rho1 = 1.
    rho2 = 1.
    lambdaE = 1.
    bE = 0.3
    Kb = 100.
    d_E = 0.25
    Kd = 500.
    deltaE = 0.1

    # decompose state
    T1, T2, T1s, T2s, V, E = s

    # compute derivatives
    tmp1 = (1. - eps1) * k1 * V * T1
    tmp2 = (1. - f * eps1) * k2 * V * T2
    out[0] = lambda1 - d1 * T1 - tmp1
    out[1] = lambda2 - d2 * T2 - tmp2
    out[2] = tmp1 - delta * T1s - m1 * E * T1s
    out[3] = tmp2 - delta * T2s - m2 * E * T2s
    out[4] = (1. - eps2) * NT * delta * (T1s + T2s) - c * V \
        - ((1. - eps1) * rho1 * k1 * T1 +
           (1. - f * eps1) * rho2 * k2 * T2) * V
    out[5] = lambdaE + bE * (T1s + T2s) / (T1s + T2s + Kb) * E \
        - d_E * (T1s + T2s) / (T1s + T2s + Kd) * E - deltaE * E

try:
    import numba
except ImportError as e:
    print("Numba acceleration unavailable, expect slow runtime.")
else:
    dsdt_ = numba.jit(
        numba.void(numba.float64[:], numba.float64[:], numba.float64, numba.float64, numba.float64),
        nopython=True, nogil=True)(dsdt_)
