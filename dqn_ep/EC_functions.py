__author__ = 'frankhe'
import numpy as np
import cPickle
import heapq
from annoy import AnnoyIndex
import image_preprocessing as ip


class Buffer(object):
    def __init__(self, size, state_dimension, frequency, n_trees=100, search_k=1000):
        self.state_dimension = state_dimension
        self.lru = np.zeros(size, np.float32)
        self.state = np.zeros((size, state_dimension), np.float32)
        self.q_return = np.zeros(size, np.float32)
        self.items = 0
        self.annoy = None
        self.last_annoy_built_time = None
        self.n_trees = n_trees
        self.search_k = search_k
        self.insert_times = 0
        self.update_frequency = frequency

    def update_annoy(self, time):
        self.annoy = AnnoyIndex(self.state_dimension)
        for i in xrange(self.items):
            self.annoy.add_item(i, self.state[i])
        self.annoy.build(self.n_trees)
        self.last_annoy_built_time = time


class DistanceNode(object):
    def __init__(self, dist, index):
        self.distance = dist
        self.index = index


class QECTable(object):
    def __init__(self, knn, state_dimension, projection_type, observation_dimension, buffer_size, num_actions, rng,
                 rebuild_frequency, n_trees, search_k):
        self.knn = knn
        self.time = 0.0  # time stamp for LRU
        self.ec_buffer = []
        self.buffer_maximum_size = buffer_size
        self.rng = rng
        for i in range(num_actions):
            self.ec_buffer.append(Buffer(buffer_size, state_dimension, rebuild_frequency, n_trees, search_k))

        # projection
        """
        I tried to make a function self.projection(state)
        but cPickle can not pickle an object that has an function attribute
        """
        self._initialize_projection_function(state_dimension, observation_dimension, projection_type)

    def _initialize_projection_function(self, dimension_result, dimension_observation, p_type):
        if p_type == 'random':
            self.matrix_projection = self.rng.randn(dimension_result, dimension_observation).astype(np.float32)
        elif p_type == 'VAE':
            pass
        else:
            raise ValueError('unrecognized projection type')

    """estimate the value of Q_EC(s,a)  O(N*logK*D)  check existence: O(N) -> KNN: O(D*N*logK)"""
    def estimate(self, s, a):
        if type(a) == np.ndarray:
            a = a[0]
        state = np.dot(self.matrix_projection, s.flatten())
        self.time += 0.001

        buffer_a = self.ec_buffer[a]

        #  first check if we already have this state

        if buffer_a.annoy is None:
            # there is no approximate tree now
            for i in xrange(buffer_a.items):
                if self._similar_state(buffer_a.state[i], state):
                    buffer_a.lru[i] = self.time
                    return buffer_a.q_return[i]
        else:
            nearest_item = buffer_a.annoy.get_nns_by_vector(state, 1, buffer_a.search_k)[0]
            if self._similar_state(buffer_a.state[nearest_item], state):
                buffer_a.lru[nearest_item] = self.time
                return buffer_a.q_return[nearest_item]

        #  second KNN to evaluate the novel state

        if buffer_a.annoy is None:
            # there is no approximate tree now
            distance_list = []
            for i in xrange(buffer_a.items):
                distance_list.append(DistanceNode(self._calc_distance(state, buffer_a.state[i]), i))

            smallest = heapq.nsmallest(self.knn, distance_list, key=lambda x: x.distance)
            value = 0.0
            for d_node in smallest:
                index = d_node.index
                value += buffer_a.q_return[index]
                buffer_a.lru[index] = self.time
            return value / self.knn
        else:
            smallest = buffer_a.annoy.get_nns_by_vector(state, self.knn, buffer_a.search_k)
            value = 0.0
            for i in smallest:
                value += buffer_a.q_return[i]
                #  if this node does not change after last annoy
                if buffer_a.lru[i] <= buffer_a.last_annoy_built_time:
                    buffer_a.lru[i] = self.time
            return value / self.knn

    @staticmethod
    def _calc_distance(a, b):
        return np.sum(np.absolute(a-b))

    @staticmethod
    def _similar_state(a, b, threshold=200.0):
        dist = QECTable._calc_distance(a, b)
        if dist < threshold:
            return True
        else:
            return False

    """update Q_EC(s,a)"""
    def update(self, s, a, r):  # s is 84*84*3;  a is 0 to num_actions; r is reward
        if type(a) == np.ndarray:
            a = a[0]
        state = np.dot(self.matrix_projection, s.flatten())
        self.time += 0.001

        buffer_a = self.ec_buffer[a]

        #  first check if we already have this state

        if buffer_a.annoy is None:
            # there is no approximate tree now
            for i in xrange(buffer_a.items):
                if self._similar_state(buffer_a.state[i], state):
                    buffer_a.q_return[i] = np.maximum(buffer_a.q_return[i], r)
                    buffer_a.lru[i] = self.time
                    return
        else:
            nearest_item = buffer_a.annoy.get_nns_by_vector(state, 1, buffer_a.search_k)[0]
            if self._similar_state(buffer_a.state[nearest_item], state):
                buffer_a.q_return[nearest_item] = np.maximum(buffer_a.q_return[nearest_item], r)
                buffer_a.lru[nearest_item] = self.time
                return

        # second insert a new node
        if buffer_a.items < self.buffer_maximum_size:
            i = buffer_a.items
            buffer_a.items += 1
        else:
            i = np.argmin(buffer_a.lru)
        buffer_a.insert_times += 1
        buffer_a.state[i] = state
        buffer_a.q_return[i] = r
        buffer_a.lru[i] = self.time

        if buffer_a.annoy is None:
            if buffer_a.insert_times == self.buffer_maximum_size:
                buffer_a.update_annoy(self.time)
                buffer_a.insert_times = 0
        else:
            if buffer_a.insert_times % buffer_a.update_frequency == 0:
                buffer_a.update_annoy(self.time)


class TraceNode(object):
    def __init__(self, observation, action, reward, terminal):
        self.image = observation
        self.action = action
        self.reward = reward
        self.terminal = terminal


class TraceRecorder(object):
    def __init__(self):
        self.trace_list = []

    def add_trace(self, observation, action, reward, terminal):
        node = TraceNode(observation, action, reward, terminal)
        self.trace_list.append(node)




