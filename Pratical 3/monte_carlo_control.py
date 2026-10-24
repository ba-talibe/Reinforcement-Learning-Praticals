import numpy as np
from ForTP3 import *
import pickle



def process_episode(episode_t):
    state, action, reward, next_state = episode_t
    state = (int(state[0]), int(state[1]))  
    state = states.index(state)  
    action = actions.index(action)  
    next_state = (int(next_state[0]), int(next_state[1]))  
    next_state = states.index(next_state)  
    return state, action, reward, next_state

def generate_valide_episode(init_states, actions, Maze, exit_state, max_iter=1000):
    valid_episode = False
    while not valid_episode:
        try:
            init_state = init_states[np.random.randint(len(init_states))]
            episode = generate_episode(init_state, actions, Maze, exit_state, itermax=max_iter)
            valid_episode = True
            print(".", end="")
        except IndexError:
            continue
    return episode

def generate_valide_episode_from_policy(init_state, actions, Maze, exit_state, pi, max_iter=1000):
    valid_episode = False
    while not valid_episode:
        try:
            episode = generate_episode_from_policy(init_state, actions, Maze, exit_state, pi, itermax=max_iter)
            valid_episode = True
            print(".", end="")
        except IndexError:
            continue
    return episode

def first_visit_prediction(Maze, states, max_iter=1000):
    # episode is a list of (state, action, reward, next_state)
    # V : ndarray of shape (n_states, ) containing the value function
    V = np.zeros(len(states))  # Initialize V(s) to zero for all states
    returns = {state: [] for state in range(len(states))}  # Initialize returns for each state
    states_visited = set()
    for _ in range(max_iter):
        episode = generate_valide_episode(init_states, actions, Maze, exit_state, max_iter=max_iter)
        episode = list(map(process_episode, episode)) 
        for t in range(len(episode)):
            state, action, reward, next_state = episode[t]
            if state not in states_visited:
                states_visited.add(state)
                continue
            returns[state].append(reward)
            V[state] = np.mean(returns[state])  # Update V(s) to the average of returns for that state
    return V



def monte_carlo_exploring(maze, actions, iter_max=1000):
    states, exit_state, init_states = get_states_from_Maze(maze)
    pi = np.zeros(len(states), dtype=int)  # Initialize policy to always take action 0
    q_s_a = np.zeros((len(states), len(actions)))  # Initialize Q(s,a) to zero
    returns = {}  # Count of returns for each (s,a) pair

    for _ in range(iter_max):
        episode = generate_valide_episode(init_states, actions, maze, exit_state, max_iter=iter_max)
        episode = list(map(process_episode, episode))  # Process the episode to get indices

        episode_states = set()
        for t in reversed(range(len(episode))):
            state, action, reward, next_state = episode[t]

            episode_states.add(state)
            if (state, action) not in returns:
                returns[(state, action)] = []
            returns[(state, action)] += [reward]
            q_s_a[state][action] = np.mean(returns[(state, action)])

        for s in episode_states:
            pi[s] = np.argmax(q_s_a[s])

    return pi, q_s_a

def monte_carlo_fv_on_policy(maze, actions,epsilon, iter_max=1000):
    states, exit_state, init_states = get_states_from_Maze(maze)
    pi = np.zeros((len(states), len(actions)), dtype=float)  # Initialize policy to always take action 0
    q_s_a = np.zeros((len(states), len(actions)))  # Initialize Q(s,a) to zero
    returns = {}  # Count of returns for each (s,a) pair

    for s in range(len(states)):
        for a in range(len(actions)):
            q_s_a[s][a] = np.random.rand() 
            returns[(s, a)] = [] 
    # a revoir on doit faire un epsilon soft sur les action possible pour chaque etat
    # ou sur les actions
    for s in range(len(states)):
        pi[s] = np.random.rand(len(actions))  
        pi[s] /= np.sum(pi[s])  
       

    for _ in range(iter_max):
        init_state = init_states[np.random.randint(len(init_states))]
        episode = generate_valide_episode_from_policy(init_state, actions, maze, exit_state, pi)
        episode = list(map(process_episode, episode))  # Process the episode to get indices

        episode_states = set()
        for t in reversed(range(len(episode))):
            state, action, reward, next_state = episode[t]
            episode_states.add(state)

            if (state, action) not in returns:
                returns[(state, action)] = []
            returns[(state, action)] += [reward]
            q_s_a[state][action] = np.mean(returns[(state, action)])

        # Update policy to be epsilon-soft with respect to Q(s,a)
        for s in episode_states:
            best_action = np.argmax(q_s_a[s])
            for a in range(len(actions)):
                if a == best_action:
                    pi[s][a] = 1 - epsilon + epsilon / len(actions)  # 90% for the best action
                else:
                    pi[s][a] = epsilon / len(actions)  # 10% distributed among other actions

    return pi, q_s_a

        

if __name__ == '__main__':

    
    try:
        with open('maze.pkl', 'rb') as f:
            Maze = pickle.load(f)
    except FileNotFoundError:
        # create a new maze if the file does not exist
        app = App()
        app.mainloop()
        Maze=app.A

        # Save the generated maze
        with open('maze.pkl', 'wb') as f:
            pickle.dump(Maze, f)





    states,exit_state,init_states = get_states_from_Maze(Maze)
    init_state = init_states[np.random.randint(len(init_states))]


    print("Monte Carlo Control :")
    states,exit_state,init_states = get_states_from_Maze(Maze)
    states = list(map(lambda x: (int(x[0]), int(x[1])), states))  # Convert states to a list of tuples for consistent indexing
    init_state = init_states[np.random.randint(len(init_states))]
    try:
        with open('episode.pkl', 'rb') as f:
            episode = pickle.load(f)
    except FileNotFoundError:
        # create a new episode if the file does not exist
        episode = generate_valide_episode(init_states, actions, Maze, exit_state, max_iter=100*len(states))

    # Save the generated episode
    with open('episode.pkl', 'wb') as f:
        pickle.dump(episode, f)

    print(states.index((1, 1)))

    
    # print("first_visit_prediction :")
    # V = first_visit_prediction(Maze, states)
    # print(V)

    # print("monte_carlo_exploring :")
    # pi, q_s_a = monte_carlo_exploring(Maze, actions, iter_max=1000)
    # print("Policy (pi):")
    # print(pi)   

    print("monte_carlo_control first visit :")

    pi, q_s_a = monte_carlo_fv_on_policy(Maze, actions, epsilon=0.1, iter_max=1000)
    print("Policy (pi):")
    print(pi)
    print("Q(s,a):")
    print(q_s_a)

