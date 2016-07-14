__author__ = 'frankhe'
import numpy as np
import cPickle
import heapq
import image_preprocessing as ip


class Node(object):
    def __init__(self, time, state, q_return):
        self.lru_time = time  # time stamp used for LRU
        self.state = state
        self.QEC_value = q_return


class DistanceNode(object):
    def __init__(self, distance, index):
        self.distance = distance
        self.index = index


class QECTable(object):
    def __init__(self, knn, state_dimension, projection_type, observation_dimension, buffer_size, num_actions, rng):
        self.knn = knn
        self.time = 0.0  # time stamp for LRU
        self.ec_buffer = []
        self.buffer_maximum_size = buffer_size
        self.rng = rng
        for i in range(num_actions):
            cache = []
            self.ec_buffer.append(cache)

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
        state = np.dot(self.matrix_projection, s.flatten())
        self.time += 0.001

        for node in self.ec_buffer[a]:
            if np.allclose(node.state, state):
                # Q(s,a) already existed
                node.lru_time = self.time
                return node.QEC_value

        # KNN to evaluate the novel state
        distance_list = []
        for i in range(len(self.ec_buffer[a])):
            distance_list.append(DistanceNode(self._calc_distance(state, self.ec_buffer[a][i].state), i))
        # print_distance_list(distance_list)

        # use nsmallest so I do not need to write a heap myself
        smallest = heapq.nsmallest(self.knn, distance_list, key=lambda x: x.distance)
        # print_distance_list(smallest)

        value = 0.0
        for d_node in smallest:
            index = d_node.index
            value += self.ec_buffer[a][index].QEC_value
            self.ec_buffer[a][index].lru_time = self.time

        return value / self.knn

    @staticmethod
    def _calc_distance(a, b):
        return np.sum(a-b)

    """update Q_EC(s,a)  O(N)  check_existence: O(N) -> insert: O(1) || LRU_insert: O(N) || heap_LRU_insert: O(logN)"""
    def update(self, s, a, r):  # s is 84*84*3;  a is 0 to num_actions; r is reward
        state = np.dot(self.matrix_projection, s.flatten())
        self.time += 0.001

        for node in self.ec_buffer[a]:
            if np.allclose(node.state, state):
                # Q(s,a) already existed
                node.QEC_value = np.maximum(node.QEC_value, r)
                node.lru_time = self.time
                return

        # insert a new Q(s,a)
        qec_s_a = Node(self.time, state, r)
        if len(self.ec_buffer[a]) < self.buffer_maximum_size:
            # ec_buffer still has room for new element
            self.ec_buffer[a].append(qec_s_a)
        else:
            # ec_buffer[a] needs LRU to unleash some room
            i = self._find_lru_node_index(a)
            self.ec_buffer[a][i] = qec_s_a

    """naive approach to find LRU node  O(N)"""
    def _find_lru_node_index(self, a):
        minimum_time_stamp = float('inf')
        min_time_node_index = None
        for i in range(len(self.ec_buffer[a])):
            if self.ec_buffer[a][i].lru_time < minimum_time_stamp:
                minimum_time_stamp = self.ec_buffer[a][i].lru_time
                min_time_node_index = i
        return min_time_node_index


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
        print 'action buffer of ', a, 'length=', len(action_buffer)
        a += 1
        for node in action_buffer:
            print '(', node.lru_time, node.QEC_value, ')',
        print
    print


def print_distance_list(l):
    for x in l:
        print x.distance,
    print

if __name__ == '__main__':
    images = cPickle.load(open('game_images', mode='rb'))
    images = [x.astype(np.float32)/255.0 for x in images]

    table = QECTable(2, 64, 'random', images[0].size, 3, 2, np.random.RandomState(12345))
    table.update(images[0], 0, 1)
    table.update(images[1], 0, 2)
    table.update(images[2], 0, 3)
    table.update(images[2], 1, 4)
    table.update(images[3], 1, 5)
    print_table(table)
    table.update(images[0], 0, 10)
    print_table(table)
    table.update(images[5], 0, 6)
    print_table(table)
    print table.estimate(images[10], 0)
    print_table(table)










