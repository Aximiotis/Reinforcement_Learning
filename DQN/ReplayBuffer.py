import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.filling = 0
        self.buffer = {
            'state': np.zeros((max_size, state_dim)),
            'action': np.zeros((max_size, action_dim)),
            'reward': np.zeros((max_size, 1)),
            'next_state': np.zeros((max_size, state_dim)),
            'done': np.zeros((max_size, 1))
        }

    def add(self, state, action, reward, next_state, done):

        
        if self.filling < self.max_size:

            self.buffer['state'][self.filling] = state
            self.buffer['action'][self.filling] = action
            self.buffer['reward'][self.filling] = reward
            self.buffer['next_state'][self.filling] = next_state
            self.buffer['done'][self.filling] = done
            self.filling += 1

        else:

            self.buffer['state'] = np.roll(self.buffer['state'], -1, axis=0)
            self.buffer['action'] = np.roll(self.buffer['action'], -1, axis=0)
            self.buffer['reward'] = np.roll(self.buffer['reward'], -1, axis=0)
            self.buffer['next_state'] = np.roll(self.buffer['next_state'], -1, axis=0)
            self.buffer['done'] = np.roll(self.buffer['done'], -1, axis=0)

            self.buffer['state'][-1] = state
            self.buffer['action'][-1] = action
            self.buffer['reward'][-1] = reward
            self.buffer['next_state'][-1] = next_state
            self.buffer['done'][-1] = done

        

    def sample(self, batch_size):

        if self.filling < batch_size:
            raise ValueError("Not enough samples in the buffer to sample the requested batch size.")
        else :
            indices = np.random.choice(self.filling, batch_size, replace=False)

        return indices
    
    def convert_to_array(self,indices):


        size_indices = indices.shape[0]
        array = np.zeros((size_indices,2*self.state_dim+self.action_dim+2),dtype=float)


        array[:,0:self.state_dim] = self.buffer['state'][indices]
        array[:,self.state_dim:self.state_dim+self.action_dim] = self.buffer['action'][indices]
        array[:,self.state_dim+self.action_dim : self.state_dim+self.action_dim+1] = self.buffer['reward'][indices]
        array[:,self.state_dim+self.action_dim + 1 : 2*self.state_dim+self.action_dim + 1 ] = self.buffer['next_state'][indices]
        array[:,2*self.state_dim+self.action_dim + 1 : 2*self.state_dim+self.action_dim + 2 ] = self.buffer['done'][indices]

        return  array
        

