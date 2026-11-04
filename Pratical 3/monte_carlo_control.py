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

