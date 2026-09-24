import numpy as np



def policy_evalution(pi, states, actions, transitions, discount=1):

    v = np.zeros(len(states))
    