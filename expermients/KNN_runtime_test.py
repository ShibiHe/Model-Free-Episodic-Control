__author__ = 'frankhe'
import numpy as np
import cPickle
import heapq
import time
from sklearn.neighbors import KDTree, BallTree
import kdtree
from annoy import AnnoyIndex
from pyflann import FLANN
data_num = 100000
test_num_for_each = 20

dimension_result = 64
dimension_observation = 84*84
K = 11


def create_data():
    f1 = open('states.pkl', 'w')
    f2 = open('states_for_test.pkl', 'w')
    matrix_projection = np.random.normal(
        loc=0.0, scale=1.0/np.sqrt(dimension_result), size=(dimension_result, dimension_observation))
    num = data_num
    data = np.zeros((num, dimension_result))
    for i in xrange(num):
        print "creating data", i
        data[i] = np.dot(matrix_projection, np.random.randint(0, 255, dimension_observation))
    data_states = data

    test = np.zeros((num, test_num_for_each, dimension_result))
    for i in xrange(num):
        print "creating test", i
        for j in xrange(test_num_for_each):
            test[i, j] = np.dot(matrix_projection, np.random.randint(0, 255, dimension_observation))
    test_states = test

    cPickle.dump(data_states, f1, 2)
    cPickle.dump(test_states, f2, 2)
    f1.close()
    f2.close()


class DistanceNode(object):
    def __init__(self, distance, index):
        self.distance = distance
        self.index = index


def make_test(test_start=1000, test_end=1050):
    f1 = open('states.pkl', 'r')
    f2 = open('states_for_test.pkl', 'r')
    data_states = cPickle.load(f1)
    test_states = cPickle.load(f2)
    f1.close()
    f2.close()

    time_brute = []
    time_sk_kd = []
    time_sk_ball = []
    time_kdtree = []
    time_annoy = []
    time_flann = []
    time_brute_tot = time_sk_kd_tot = time_sk_ball_tot = time_kdtree_tot = time_annoy_tot = time_flann_tot = 0

    kdtree_tree = None
    for items in xrange(test_start, test_end):
        print "item:", items

        ground_truth = np.zeros((test_num_for_each, K), dtype=np.int32)
        time_brute_start = time.time()
        for no_test in xrange(test_num_for_each):
            distance_list = []
            current_state = test_states[items, no_test]
            for target in xrange(items):
                target_state = data_states[target]
                distance_list.append(DistanceNode(np.sum(np.absolute(current_state - target_state)**2), target))
            smallest = heapq.nsmallest(K, distance_list, key=lambda x: x.distance)
            ground_truth[no_test] = [x.index for x in smallest]
        time_brute_end = time.time()
        time_brute.append(time_brute_end - time_brute_start)
        time_brute_tot += time_brute[-1]
        # print ground_truth

        time_sk_kd_start = time.time()
        tree = KDTree(data_states[:items, :])
        dist, indices = tree.query(test_states[items], K)
        time_sk_kd_end = time.time()
        time_sk_kd.append(time_sk_kd_end - time_sk_kd_start)
        time_sk_kd_tot += time_sk_kd[-1]
        # print indices

        time_sk_ball_start = time.time()
        tree = BallTree(data_states[:items, :], 10000)
        dist, indices = tree.query(test_states[items], K)
        time_sk_ball_end = time.time()
        time_sk_ball.append(time_sk_ball_end - time_sk_ball_start)
        time_sk_ball_tot += time_sk_ball[-1]
        # print indices

        """
        annoy is absolutely disappointing for its low speed and poor accuracy.
        """
        time_annoy_start = time.time()
        annoy_result = np.zeros((test_num_for_each, K), dtype=np.int32)
        tree = AnnoyIndex(dimension_result)
        for i in xrange(items):
            tree.add_item(i, data_states[i, :])
        tree.build(10)
        for no_test in xrange(test_num_for_each):
            current_state = test_states[items, no_test]
            annoy_result[no_test] = tree.get_nns_by_vector(current_state, K)
        time_annoy_end = time.time()
        time_annoy.append(time_annoy_end - time_annoy_start)
        time_annoy_tot += time_annoy[-1]
        # print annoy_result
        # print annoy_result - indices

        """
        flann is still not very ideal
        """

        time_flann_start = time.time()
        flann = FLANN()
        result, dist = flann.nn(data_states[:items, :], test_states[items], K, algorithm='kdtree', trees=10, checks=16)
        time_flann_end = time.time()
        time_flann.append(time_flann_end - time_flann_start)
        time_flann_tot += time_flann[-1]
        # print result-indices

        """
        This kdtree module is so disappointing!!!! It is 100 times slower than Sklearn and even slower than brute force,
        more over it even makes mistakes.

        This kdtree module supports online insertion and deletion. I thought it would be much faster than Sklearn
         KdTree which rebuilds the tree every time. But the truth is the opposite.
        """

        # time_kdtree_start = time.time()
        # if kdtree_tree is None:
        #     point_list = [MyTuple(data_states[i, :], i) for i in xrange(items)]
        #     kdtree_tree = kdtree.create(point_list)
        # else:
        #     point = MyTuple(data_states[items, :], items)
        #     kdtree_tree.add(point)
        # kdtree_result = np.zeros((test_num_for_each, K), dtype=np.int32)
        # for no_test in xrange(test_num_for_each):
        #     current_state = test_states[items, no_test]
        #     smallest = kdtree_tree.search_knn(MyTuple(current_state, -1), K)
        #     kdtree_result[no_test] = [x[0].data.pos for x in smallest]
        # time_kdtree_end = time.time()
        # time_kdtree.append(time_kdtree_end - time_kdtree_start)
        # time_kdtree_tot += time_kdtree[-1]
        # print kdtree_result
        # print kdtree_result-indices

    print 'brute force:', time_brute_tot
    print 'sklearn KDTree', time_sk_kd_tot
    print 'sklearn BallTree', time_sk_ball_tot
    print 'approximate annoy', time_annoy_tot
    print 'approximate flann', time_flann_tot
    print 'kdtree (deprecated)', time_kdtree_tot


class MyTuple(tuple):
    def __new__(cls, data, pos):
        return super(MyTuple, cls).__new__(cls, data)

    def __init__(self, data, pos):
        self.pos = pos

if __name__ == '__main__':
    # create_data()
    make_test()






