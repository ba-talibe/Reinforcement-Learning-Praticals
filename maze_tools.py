import numpy as np
import matplotlib.pyplot as plt 
import plotly.express as px 
import pickle

class MazeEnv:
    def __init__(self, A=None, file=None) -> None:
        if A is None and file is None:
            raise Exception("Booth A matrix and pickle file are None")
        
        self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']

        if A is not None:
            self.Maze = A
        
        if file is not None:
            try:
                with open(file, 'rb') as f:
                    self.Maze = pickle.load(f)
            except Exception as e:
                print(e)

        self.init_states_from_Maze()

        


    def init_states_from_Maze(self) :
        # init states, exit_state, init_states,  
        self.states = list(np.argwhere(self.Maze == 1))
        M = np.zeros(self.Maze.shape)
        M[:,0]=1
        M[:,-1] = 1
        M[0,:]=1
        M[-1,:]=1
        
        self.exit_state = np.argwhere(self.Maze*M)[0]
        self.init_states = list(np.copy(self.states))
        self.init_states.pop(np.argwhere((self.init_states == self.exit_state).sum(axis =1) == 2)[0][0])

    def get_actions(self, state):
        # list of possible actions 
        local_actions = list()
        for a in self.actions :
            m = [0,0]
            if a == "UP" :
                m[0] = -1   
            elif a == "DOWN" :
                m[0] = +1
            elif a == "LEFT" :
                m[1] = -1
            else : # a=="RIGHT"
                m[1] = +1
            if self.Maze[state[0]+m[0],state[1]+m[1]] == 1.0 :    
                local_actions.append(a)
        return local_actions

    
    def next_state(self, state, action,):
        # give the resulting state end reward 
        new_state = np.copy(state)
        if action == "UP" :
            new_state[0] -= 1   
        elif action == "DOWN" :
            new_state[0] += 1
        elif action == "LEFT" :
            new_state[1] -= 1
        else : # a=="RIGHT"
            new_state[1] += 1
        
        if all(new_state == self.exit_state) :
            reward = 1
        else:
            reward = 0
        return new_state, reward

