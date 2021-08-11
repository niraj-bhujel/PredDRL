import numpy as np
import random
from collections import defaultdict

from .segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size, state_dim=1, act_dim=1, rew_dim=1):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.rew_dim = rew_dim

    def __len__(self):
        return len(self._storage)

    def add(self, obs, act, reward, next_obs, done):

        data = (obs, act, reward, next_obs, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        # print(self._next_idx)

    def _encode_sample(self, idxes):

        obs, act, rew, next_obs, done = [], [], [], [], []

        for i in idxes:

            obs_, act_, rew_, next_obs_, done_ = self._storage[i]

            obs.append(obs_)
            act.append(act_)
            rew.append(rew_)
            next_obs.append(next_obs_)
            done.append(done_)

        samples = {'obs' : np.array(obs, dtype=np.float32).reshape([-1, self.state_dim]), 
                    'act': np.array(act, dtype=np.float32).reshape([-1, self.act_dim]),
                    'rew' : np.array(rew, dtype=np.float32).reshape([-1, self.rew_dim]),
                    'next_obs' : np.array(next_obs, dtype=np.float32).reshape([-1, self.state_dim]),
                    'done' : np.array(done, dtype=np.float32).reshape([-1, 1])

        }
        return samples
        # return np.array(obs), np.array(act), np.array(rewards), np.array(next_obs), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of act executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha=0.6, beta=0.4, state_dim=1, act_dim=1, rew_dim=1):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size, state_dim=state_dim, act_dim=act_dim, rew_dim=rew_dim)
        assert alpha >= 0
        self._alpha = alpha
        self._beta = beta
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
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


        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of act executed given obs_batch
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
        assert self._beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-self._beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-self._beta)
            weights.append(weight / max_weight)

        encoded_sample = self._encode_sample(idxes)

        encoded_sample['weights'] = np.array(weights, dtype=np.float32)
        encoded_sample['indexes'] = np.array(idxes, dtype=np.int32)

        return encoded_sample

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
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
