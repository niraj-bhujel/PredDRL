

import random
import numpy as np
from collections import deque, defaultdict
import heapq
from itertools import count

from .segment_tree import SegmentTree, MinSegmentTree, SumSegmentTree

'''
Adopted from https://github.com/cocolico14/N-step-Dueling-DDQN-PER-Pacman/
'''
class NstepBuffer:
    """docstring for """
    def __init__(self, size=4,  gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):

        self.n_step = size
        self.gamma = gamma 
        
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        

        self.n_step_buffer = deque([], size)
        
        self.cnt = count()
        
        self.Nstep_size = size
        self.Nstep_gamma = gamma
        self.stored_size = 0
        self.buffer_size = self.Nstep_size - 1
        
        # self.buffer = defaultdict(lambda: deque([], maxlen=self.Nstep_size))
        self.buffer = defaultdict(list)
        
    def add_(self, state, action, reward, next_state, done, td_error):
        '''
        https://github.com/cocolico14/N-step-Dueling-DDQN-PER-Pacman
        '''
        # n-step queue for calculating return of n previous steps
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_step:
          return

        l_reward, l_next_state, l_done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]

            l_reward = r + self.gamma * l_reward * (1 - d)
            l_next_state, l_done = (n_s, d) if d else (l_next_state, l_done)
        
        l_state, l_action = self.n_step_buffer[0][:2]

        t = (l_state, l_action, l_reward, l_next_state, l_done)
        return t

    def add(self, state, action, reward, next_state, done, **kwargs):
        """Add envronment into local buffer.
        Paremeters
        ----------
        **kwargs : keyword arguments
            Values to be added.
        Returns
        -------
        env : dict or None
            Values with Nstep reward calculated. When the local buffer does not
            store enough cache items, returns 'None'.
        """
        
        N = 1 # assume 1 samples now
        end = self.stored_size + N


        # Case 1
        #   If Nstep buffer don't become full, store all the input transitions.
        #   These transitions are partially calculated.
        if end <= self.buffer_size:
            
            self.buffer['state'].append(state)
            self.buffer['action'].append(action)
            self.buffer['done'].append(done)
            
            # for name, stored_b in self.buffer.items():
            #     if self.Nstep_rew is not None and np.isin(name, self.Nstep_rew).any():
            #         # Calculate later.
            #         pass
            #     elif (self.Nstep_next is not None
            #           and np.isin(name,self.Nstep_next).any()):
            #         # Do nothing.
            #         pass
            #     else:
            #         stored_b[self.stored_size:end] = self._extract(kwargs,name) # _extract return array of shape (-1, dim)

            # Nstep reward must be calculated after "done" filling
            # gamma = (1.0 - self.buffer["done"][:end]) * self.Nstep_gamma # numpy array of shape (end, 1)
            gamma = [1.0-self.buffer['done'][i]*self.Nstep_gamma for i in range(0, end)]
            
            # work on progress!
            # # if self.Nstep_rew is not None:
            #     max_slide = min(self.Nstep_size - self.stored_size, N)
            #     max_slide *= -1
            #     # for name in self.Nstep_rew:
                    
            #         # ext_b = self._extract(kwargs, name).copy() # reward array of shape (-1, 1)
            #         # self.buffer[name][self.stored_size:end] = ext_b
            #         self.buffer['reward'][self.stored_size:end] = reward
        
            #         for i in range(self.stored_size-1, max_slide, -1):
            #             stored_begin = max(i, 0)
            #             stored_end = i+N
            #             ext_begin = max(-i, 0)
            #             ext_b[ext_begin:] *= gamma[stored_begin:stored_end]
            #             self.buffer[name][stored_begin:stored_end] +=ext_b[ext_begin:]
    
            self.stored_size = end
            return None

        # Case 2
        #   If we have enough transitions, return calculated transtions
        diff_N = self.buffer_size - self.stored_size
        add_N = N - diff_N
        NisBigger = (add_N > self.buffer_size)
        end = self.buffer_size if NisBigger else add_N

        # Nstep reward must be calculated before "done" filling
        gamma = np.ones((self.stored_size + N,1),dtype=np.single)
        gamma[:self.stored_size] -= self.buffer["done"][:self.stored_size]
        gamma[self.stored_size:] -= self._extract(kwargs,"done")
        gamma *= self.Nstep_gamma
        if self.Nstep_rew is not None:
            max_slide = min(self.Nstep_size - self.stored_size,N)
            max_slide *= -1
            for name in self.Nstep_rew:
                stored_b = self.buffer[name]
                ext_b = self._extract(kwargs,name)

                copy_ext = ext_b.copy()
                if diff_N:
                    stored_b[self.stored_size:] = ext_b[:diff_N]
                    ext_b = ext_b[diff_N:]

                for i in range(self.stored_size-1,max_slide,-1):
                    stored_begin = max(i,0)
                    stored_end = i+N
                    ext_begin = max(-i,0)
                    copy_ext[ext_begin:] *= gamma[stored_begin:stored_end]
                    if stored_end <= self.buffer_size:
                        stored_b[stored_begin:stored_end] += copy_ext[ext_begin:]
                    else:
                        spilled_N = stored_end - self.buffer_size
                        stored_b[stored_begin:] += copy_ext[ext_begin:-spilled_N]
                        ext_b[:spilled_N] += copy_ext[-spilled_N:]

                self._roll(stored_b,ext_b,end,NisBigger,kwargs,name,add_N)

        for name, stored_b in self.buffer.items():
            if self.Nstep_rew is not None and np.isin(name,self.Nstep_rew).any():
                # Calculated.
                pass
            elif (self.Nstep_next is not None
                  and np.isin(name,self.Nstep_next).any()):
                kwargs[name] = self._extract(kwargs,name)[diff_N:]
            else:
                ext_b = self._extract(kwargs,name)

                if diff_N:
                    stored_b[self.stored_size:] = ext_b[:diff_N]
                    ext_b = ext_b[diff_N:]

                self._roll(stored_b,ext_b,end,NisBigger,kwargs,name,add_N)

        done = kwargs["done"]

        for i in range(1,self.buffer_size):
            if i <= add_N:
                done[:-i] += kwargs["done"][i:]
                done[-i:] += self.buffer["done"][:i]
            else:
                done += self.buffer["done"][i-add_N:i]

        self.stored_size = self.buffer_size
        return kwargs



'''
Adopted from https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/component/replay.py
'''

class ReplayBuffer:
    def __init__(self, size, use_nstep=False, n_step=4, gamma=0.995):

        self._storage = []
        self._maxsize = size
        if use_nstep:
            self.n_step_buffer = NstepBuffer(size=n_step, gamma=gamma)

    def add(self, transition):
        self._storage.append(transition)
        if len(self._storage) > self._maxsize:
            del self.memory[0]

    def _encode_sample(self, idxes):
        return [self._storage[i] for i in idxes]

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes), None, None

    def __len__(self):
        return len(self._storage)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha=0.6, beta_start=0.4, beta_frames=100000, use_nstep=False, n_step=4):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size=size)

        self._next_idx = 0

        assert alpha >= 0
        self._alpha = alpha

        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame=1

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        
        self._max_priority = 1.0

        # if use_nstep:
        #     self.n_step_buffer = NstepBuffer(size=n_step)

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def add(self, data):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data

        self._next_idx = (self._next_idx + 1) % self._maxsize


        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        batch_size = min(batch_size, len(self._storage))
        idxes = self._sample_proportional(batch_size)

        weights = []

        #find smallest sampling prob: p_min = smallest priority^alpha / sum of priorities^alpha
        p_min = self._it_min.min() / self._it_sum.sum()

        beta = self.beta_by_frame(self.frame)
        self.frame+=1
        
        #max_weight given to smallest prob
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        # weights = torch.tensor(weights, device=device, dtype=torch.float)
        weights = np.array(weights, dtype=np.float32)
        idxes = np.array(idxes, dtype=np.int32)
        encoded_sample = self._encode_sample(idxes)

        return encoded_sample, idxes, weights

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert 0 <= idx < len(self._storage)
            assert priority >0

            self._it_sum[idx] = (priority+1e-5) ** self._alpha
            self._it_min[idx] = (priority+1e-5) ** self._alpha

            self._max_priority = max(self._max_priority, (priority+1e-5))


class RecurrentExperienceReplayMemory:
    def __init__(self, capacity, sequence_length=10):
        self.capacity = capacity
        self.memory = []
        self.seq_length=sequence_length

    def add(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        finish = random.sample(range(0, len(self.memory)), batch_size)
        begin = [x-self.seq_length for x in finish]
        samp = []
        for start, end in zip(begin, finish):
            #correct for sampling near beginning
            final = self.memory[max(start+1,0):end+1]
            
            #correct for sampling across episodes
            for i in range(len(final)-2, -1, -1):
                if final[i][3] is None:
                    final = final[i+1:]
                    break
                    
            #pad beginning to account for corrections
            while(len(final)<self.seq_length):
                final = [(np.zeros_like(self.memory[0][0]), 0, 0, np.zeros_like(self.memory[0][3]))] + final
                            
            samp+=final

        #returns flattened version
        return samp, None, None

    def __len__(self):
        return len(self.memory)


if __name__=='__main__':
    nstep_buffer = NstepPrioritizedBuffer()
    for i in range(10):
        nstep_buffer.add(state=np.random.random((24,)), 
                         action=np.random.random((2)), 
                         reward=np.random.random((1)), 
                         next_state=np.random.random((24,)), 
                         td_error=np.random.random((1,)), 
                         done=np.random.randint(0, 1, size=(1)))