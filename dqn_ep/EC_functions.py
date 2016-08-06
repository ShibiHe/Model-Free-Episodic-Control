__author__ = 'frankhe'
import numpy as np
import cPickle
import heapq
"""
due to the KNN_runtime_test.py  we use BallTree
"""
from sklearn.neighbors import BallTree
import image_preprocessing as ip
import logging


class Buffer(object):
    def __init__(self, size, state_dimension, frequency):
        self.size = size
        self.state_dimension = state_dimension
        self.lru = np.zeros(size, np.float32)
        self.state = np.zeros((size, state_dimension), np.float32)
        self.q_return = np.zeros(size, np.float32)
        self.items = 0
        self.tree = None
        self.last_tree_built_time = None
        self.insert_times = 0
        self.update_frequency = frequency
        self.mini_frequency = 1000

    def update_tree(self, time):
        print 'rebuild tree'
        self.tree = BallTree(self.state[:self.items, :], leaf_size=self.size)
        self.last_tree_built_time = time
        print 'rebuild done'


class DistanceNode(object):
    def __init__(self, dist, index):
        self.distance = dist
        self.index = index


class QECTable(object):
    def __init__(self, knn, state_dimension, projection_type, observation_dimension, buffer_size, num_actions, rng,
                 rebuild_frequency):
        self.knn = knn
        self.time = 0.0  # time stamp for LRU
        self.ec_buffer = []
        self.buffer_maximum_size = buffer_size
        self.rng = rng
        for i in range(num_actions):
            self.ec_buffer.append(Buffer(buffer_size, state_dimension, rebuild_frequency))

        # projection
        """
        I tried to make a function self.projection(state)
        but cPickle can not pickle an object that has an function attribute
        """
        self._initialize_projection_function(state_dimension, observation_dimension, projection_type)

    def _initialize_projection_function(self, dimension_result, dimension_observation, p_type):
        if p_type == 'random':
            self.matrix_projection = self.rng.normal(loc=0.0, scale=1.0/np.sqrt(dimension_result),
                                                     size=(dimension_result, dimension_observation)).astype(np.float32)
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

        if buffer_a.tree is None:
            # there is no knn tree now
            for i in xrange(buffer_a.items):
                if self._similar_state(buffer_a.state[i], state):
                    buffer_a.lru[i] = self.time
                    return buffer_a.q_return[i]
        else:
            nearest_item = buffer_a.tree.query(state.reshape((1, -1)), return_distance=False)[0]
            if self._similar_state(buffer_a.state[nearest_item], state):
                buffer_a.lru[nearest_item] = self.time
                return buffer_a.q_return[nearest_item]

        #  second KNN to evaluate the novel state

        if buffer_a.tree is None:
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
            smallest = buffer_a.tree.query(state.reshape((1, -1)), k=self.knn, return_distance=False)
            value = 0.0
            for i in smallest:
                value += buffer_a.q_return[i]
                #  if this node does not change after last annoy
                if buffer_a.lru[i] <= buffer_a.last_tree_built_time:
                    buffer_a.lru[i] = self.time
            return value / self.knn

    @staticmethod
    def _calc_distance(a, b):
        return np.sum(np.absolute(a-b)**2)

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

        if buffer_a.tree is None:
            # there is no approximate tree now
            for i in xrange(buffer_a.items):
                if self._similar_state(buffer_a.state[i], state):
                    buffer_a.q_return[i] = np.maximum(buffer_a.q_return[i], r)
                    buffer_a.lru[i] = self.time
                    return
        else:
            nearest_item = buffer_a.tree.query(state.reshape((1, -1)), return_distance=False)[0]
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
        # logging.info('insert at {}'.format(buffer_a.insert_times))
        buffer_a.insert_times += 1
        buffer_a.state[i] = state
        buffer_a.q_return[i] = r
        buffer_a.lru[i] = self.time

        if buffer_a.tree is None:
            if buffer_a.insert_times == np.minimum(buffer_a.mini_frequency, self.buffer_maximum_size):
                buffer_a.update_tree(self.time)
                buffer_a.insert_times = 0
        else:
            if buffer_a.insert_times % buffer_a.mini_frequency == 0:
                buffer_a.update_tree(self.time)
                buffer_a.mini_frequency = np.minimum(buffer_a.mini_frequency+1, buffer_a.update_frequency)


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


def print_table(table):
    a = 0
    for action_buffer in table.ec_buffer:
        print 'action buffer of ', a, 'length=', action_buffer.items
        a += 1
        for i in range(action_buffer.items):
            print '(', action_buffer.lru[i], action_buffer.q_return[i], ')',
        print
    print

if __name__ == '__main__':
    images = cPickle.load(open('game_images', mode='rb'))
    images = [x.astype(np.float32)/255.0 for x in images]
    # from images2gif import writeGif
    # writeGif('see.gif', images)

    table = QECTable(2, 64, 'random', images[0].size, 10, 2, np.random.RandomState(12345), 10, 1, 10)
    # distance = np.zeros((100, 100), np.float32)
    # for i in range(100):
    #     for j in range(i+1, 100):
    #         distance[i, j] = distance[j, i] = QECTable._calc_distance(images[i], images[j])
    # np.set_printoptions(threshold=np.inf)
    # print distance
    #
    # raw_input()
    table.update(images[0], 0, 1)
    table.update(images[1], 0, 2)
    table.update(images[2], 0, 3)
    table.update(images[3], 0, 4)
    table.update(images[4], 0, 5)
    table.update(images[5], 0, 6)
    table.update(images[6], 0, 7)
    table.update(images[33], 0, 8)
    table.update(images[34], 0, 9)
    table.update(images[35], 0, 10)
    table.update(images[36], 0, 11)
    print_table(table)
    table.update(images[37], 0, 12)
    table.update(images[38], 0, 13)
    print_table(table)


