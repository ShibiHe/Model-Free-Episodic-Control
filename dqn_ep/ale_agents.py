__author__ = 'frankhe'
"""
episodic control and DQN agents
"""
import time
import os
import logging
import numpy as np
import cPickle
import EC_functions
from images2gif import writeGif

import ale_data_set
import sys
sys.setrecursionlimit(10000)


class EpisodicControl(object):
    def __init__(self, qec_table, ec_discount, num_actions, epsilon_start,
                 epsilon_min, epsilon_decay, exp_pref, ec_testing, rng):
        self.qec_table = qec_table
        self.ec_discount = ec_discount
        self.num_actions = num_actions
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.exp_pref = exp_pref
        self.rng = rng

        self.trace_list = EC_functions.TraceRecorder()

        self.epsilon = self.epsilon_start
        if self.epsilon_decay != 0:
            self.epsilon_rate = ((self.epsilon_start - self.epsilon_min) /
                                 self.epsilon_decay)
        else:
            self.epsilon_rate = 0

        # CREATE A FOLDER TO HOLD RESULTS
        time_str = time.strftime("_%m-%d-%H-%M_", time.gmtime())
        self.exp_dir = self.exp_pref + time_str + \
                       "{}".format(self.ec_discount).replace(".", "p")

        try:
            os.stat(self.exp_dir)
        except OSError:
            os.makedirs(self.exp_dir)

        self._open_results_file()

        self.step_counter = None
        self.episode_reward = None

        self.total_reward = 0.
        self.total_episodes = 0

        self.start_time = None

        self.last_img = None
        self.last_action = None

        self.steps_sec_ema = 0.

        self.play_images = []
        self.testing = ec_testing

        self.program_start_time = None
        self.last_count_time = None

    def time_count_start(self):
        self.last_count_time = self.program_start_time = time.time()

    def _open_results_file(self):
        logging.info("OPENING " + self.exp_dir + '/results.csv')
        self.results_file = open(self.exp_dir + '/results.csv', 'w', 0)
        self.results_file.write('epoch, episode_nums, total_reward, avg_reward, epoch time, total time\n')
        self.results_file.flush()

    def _update_results_file(self, epoch, total_episodes, total_reward):
        this_time = time.time()
        total_time = this_time-self.program_start_time
        epoch_time = this_time-self.last_count_time
        out = "{},{},{},{},{},{}\n".format(epoch, total_episodes, total_reward, total_reward/total_episodes,
                                           epoch_time, total_time)
        self.last_count_time = this_time
        self.results_file.write(out)
        self.results_file.flush()

    def start_episode(self, observation):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - height x width numpy array

        Returns:
           An integer action
        """

        self.step_counter = 0
        self.episode_reward = 0

        self.trace_list.trace_list = []
        self.start_time = time.time()
        return_action = self.rng.randint(0, self.num_actions)

        self.last_action = return_action

        self.last_img = observation

        return return_action

    def _choose_action(self, trace_list, qec_table, epsilon, observation, reward):
        trace_list.add_trace(self.last_img, self.last_action, reward, False)

        # epsilon greedy
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)

        value = -100
        maximum_action = 0
        # argmax(Q(s,a))
        for action in range(self.num_actions):
            value_t = qec_table.estimate(observation, action)
            if value_t > value:
                value = value_t
                maximum_action = action

        return maximum_action

    def step(self, reward, observation):
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - A height x width numpy array

        Returns:
           An integer action.

        """

        self.step_counter += 1
        self.episode_reward += reward

        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_rate)

        action = self._choose_action(self.trace_list, self.qec_table, self.epsilon, observation, np.clip(reward, -1, 1))

        self.last_action = action
        self.last_img = observation

        return action

    def end_episode(self, reward, terminal=True):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.
           terminal    - Whether the episode ended intrinsically
                         (ie we didn't run out of steps)
        Returns:
            None
        """

        self.episode_reward += reward
        self.total_reward += self.episode_reward
        self.total_episodes += 1
        self.step_counter += 1
        total_time = time.time() - self.start_time

        # Store the latest sample.
        self.trace_list.add_trace(self.last_img, self.last_action, np.clip(reward, -1, 1), True)

        # calculate time
        rho = 0.98
        self.steps_sec_ema *= rho
        self.steps_sec_ema += (1. - rho) * (self.step_counter/total_time)
        logging.info("steps/second: {:.2f}, avg: {:.2f}".format(
            self.step_counter/total_time, self.steps_sec_ema))
        logging.info('episode {} reward: {:.2f}'.format(self.total_episodes, self.episode_reward))

        if self.testing:
            for node in self.trace_list.trace_list:
                self.play_images.append(node.image)
            # skip the update process
            return

        """
        do update
        """
        q_return = 0.
        # last_q_return = -1.0
        for i in range(len(self.trace_list.trace_list)-1, -1, -1):
            node = self.trace_list.trace_list[i]
            q_return = q_return * self.ec_discount + node.reward
            self.qec_table.update(node.image, node.action, q_return)
            # if not np.isclose(q_return, last_q_return):
            #     self.qec_table.update(node.image, node.action, q_return)
            #     last_q_return = q_return

    def finish_epoch(self, epoch):
        # so large that i only keep one
        qec_file = open(self.exp_dir + '/qec_table_file_' + \
                        '.pkl', 'w')
        cPickle.dump(self.qec_table, qec_file, 2)
        qec_file.close()

        self._update_results_file(epoch, self.total_episodes, self.total_reward)
        self.total_episodes = 0
        self.total_reward = 0

        # EC_functions.print_table(self.qec_table)

        if self.testing:
            writeGif(self.exp_dir + '/played.gif', self.play_images)

            # handle = open('played_images', 'w+')
            # cPickle.dump(self.play_images, handle, 2)


class NeuralAgent(object):

    def __init__(self, q_network, epsilon_start, epsilon_min,
                 epsilon_decay, replay_memory_size, exp_pref,
                 replay_start_size, update_frequency, rng):

        self.network = q_network
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_memory_size = replay_memory_size
        self.exp_pref = exp_pref
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.rng = rng

        self.phi_length = self.network.num_frames
        self.image_width = self.network.input_width
        self.image_height = self.network.input_height

        # CREATE A FOLDER TO HOLD RESULTS
        time_str = time.strftime("_%m-%d-%H-%M_", time.gmtime())
        self.exp_dir = self.exp_pref + time_str + \
                       "{}".format(self.network.lr).replace(".", "p") + "_" \
                       + "{}".format(self.network.discount).replace(".", "p")

        try:
            os.stat(self.exp_dir)
        except OSError:
            os.makedirs(self.exp_dir)

        self.num_actions = self.network.num_actions

        self.data_set = ale_data_set.DataSet(width=self.image_width,
                                             height=self.image_height,
                                             rng=rng,
                                             max_steps=self.replay_memory_size,
                                             phi_length=self.phi_length)

        # just needs to be big enough to create phi's
        self.test_data_set = ale_data_set.DataSet(width=self.image_width,
                                                  height=self.image_height,
                                                  rng=rng,
                                                  max_steps=self.phi_length * 2,
                                                  phi_length=self.phi_length)
        self.epsilon = self.epsilon_start
        if self.epsilon_decay != 0:
            self.epsilon_rate = ((self.epsilon_start - self.epsilon_min) /
                                 self.epsilon_decay)
        else:
            self.epsilon_rate = 0

        self.testing = False

        self._open_results_file()
        self._open_learning_file()

        self.episode_counter = 0
        self.batch_counter = 0

        self.holdout_data = None

        # In order to add an element to the data set we need the
        # previous state and action and the current reward.  These
        # will be used to store states and actions.
        self.last_img = None
        self.last_action = None

        # Exponential moving average of runtime performance.
        self.steps_sec_ema = 0.

        self.program_start_time = None
        self.last_count_time = None
        self.epoch_time = None
        self.total_time = None

    def time_count_start(self):
        self.last_count_time = self.program_start_time = time.time()

    def _open_results_file(self):
        logging.info("OPENING " + self.exp_dir + '/results.csv')
        self.results_file = open(self.exp_dir + '/results.csv', 'w', 0)
        self.results_file.write(\
            'epoch,num_episodes,total_reward,reward_per_epoch,mean_q, epoch time, total time\n')
        self.results_file.flush()

    def _open_learning_file(self):
        self.learning_file = open(self.exp_dir + '/learning.csv', 'w', 0)
        self.learning_file.write('mean_loss,epsilon\n')
        self.learning_file.flush()

    def _update_results_file(self, epoch, num_episodes, holdout_sum):
        out = "{},{},{},{},{},{},{}\n".format(epoch, num_episodes, self.total_reward,
                                        self.total_reward / float(num_episodes),
                                        holdout_sum, self.epoch_time, self.total_time)
        self.last_count_time = time.time()
        self.results_file.write(out)
        self.results_file.flush()

    def _update_learning_file(self):
        out = "{},{}\n".format(np.mean(self.loss_averages),
                               self.epsilon)
        self.learning_file.write(out)
        self.learning_file.flush()

    def start_episode(self, observation):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - height x width numpy array

        Returns:
           An integer action
        """

        self.step_counter = 0
        self.batch_counter = 0
        self.episode_reward = 0

        # We report the mean loss for every epoch.
        self.loss_averages = []

        self.start_time = time.time()
        return_action = self.rng.randint(0, self.num_actions)

        self.last_action = return_action

        self.last_img = observation

        return return_action

    def _show_phis(self, phi1, phi2):
        import matplotlib.pyplot as plt
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+1)
            plt.imshow(phi1[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+5)
            plt.imshow(phi2[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        plt.show()

    def step(self, reward, observation):
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - A height x width numpy array

        Returns:
           An integer action.

        """

        self.step_counter += 1

        # TESTING---------------------------
        if self.testing:
            self.episode_reward += reward
            action = self._choose_action(self.test_data_set, .05,
                                         observation, np.clip(reward, -1, 1))

        # NOT TESTING---------------------------
        else:

            if len(self.data_set) > self.replay_start_size:
                self.epsilon = max(self.epsilon_min,
                                   self.epsilon - self.epsilon_rate)

                action = self._choose_action(self.data_set, self.epsilon,
                                             observation,
                                             np.clip(reward, -1, 1))

                if self.step_counter % self.update_frequency == 0:
                    loss = self._do_training()
                    self.batch_counter += 1
                    self.loss_averages.append(loss)

            else:  # Still gathering initial random data...
                action = self._choose_action(self.data_set, self.epsilon,
                                             observation,
                                             np.clip(reward, -1, 1))

        self.last_action = action
        self.last_img = observation

        return action

    def _choose_action(self, data_set, epsilon, cur_img, reward):
        """
        Add the most recent data to the data set and choose
        an action based on the current policy.
        """

        data_set.add_sample(self.last_img, self.last_action, reward, False)
        if self.step_counter >= self.phi_length:
            phi = data_set.phi(cur_img)
            action = self.network.choose_action(phi, epsilon)
        else:
            action = self.rng.randint(0, self.num_actions)

        return action

    def _do_training(self):
        """
        Returns the average loss for the current batch.
        May be overridden if a subclass needs to train the network
        differently.
        """
        imgs, actions, rewards, terminals = \
                                self.data_set.random_batch(
                                    self.network.batch_size)
        return self.network.train(imgs, actions, rewards, terminals)

    def end_episode(self, reward, terminal=True):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.
           terminal    - Whether the episode ended intrinsically
                         (ie we didn't run out of steps)
        Returns:
            None
        """

        self.episode_reward += reward
        self.step_counter += 1
        total_time = time.time() - self.start_time

        if self.testing:
            # If we run out of time, only count the last episode if
            # it was the only episode.
            if terminal or self.episode_counter == 0:
                self.episode_counter += 1
                self.total_reward += self.episode_reward
        else:

            # Store the latest sample.
            self.data_set.add_sample(self.last_img,
                                     self.last_action,
                                     np.clip(reward, -1, 1),
                                     True)

            rho = 0.98
            self.steps_sec_ema *= rho
            self.steps_sec_ema += (1. - rho) * (self.step_counter/total_time)

            logging.info("steps/second: {:.2f}, avg: {:.2f}".format(
                self.step_counter/total_time, self.steps_sec_ema))

            if self.batch_counter > 0:
                self._update_learning_file()
                logging.info("average loss: {:.4f}".format(\
                                np.mean(self.loss_averages)))

    def finish_epoch(self, epoch):
        net_file = open(self.exp_dir + '/network_file_' + str(epoch) + \
                        '.pkl', 'w')
        cPickle.dump(self.network, net_file, -1)
        net_file.close()
        this_time = time.time()
        self.total_time = this_time-self.program_start_time
        self.epoch_time = this_time-self.last_count_time

    def start_testing(self):
        self.testing = True
        self.total_reward = 0
        self.episode_counter = 0

    def finish_testing(self, epoch):
        self.testing = False
        holdout_size = 3200

        if self.holdout_data is None and len(self.data_set) > holdout_size:
            imgs, _, _, _ = self.data_set.random_batch(holdout_size)
            self.holdout_data = imgs[:, :self.phi_length]

        holdout_sum = 0
        if self.holdout_data is not None:
            for i in range(holdout_size):
                holdout_sum += np.max(
                    self.network.q_vals(self.holdout_data[i]))

        self._update_results_file(epoch, self.episode_counter,
                                  holdout_sum / holdout_size)


class EC_DQN(object):
    def __init__(self, q_network, qec_table, epsilon_start, epsilon_min,
                 epsilon_decay, replay_memory_size, exp_pref,
                 replay_start_size, update_frequency, ec_discount, num_actions, ec_testing, rng):
        self.network = q_network
        self.qec_table = qec_table
        self.ec_testing = ec_testing
        self.ec_discount = ec_discount
        self.num_actions = num_actions
        self.epsilon_start = epsilon_start

        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_memory_size = replay_memory_size
        self.exp_pref = exp_pref
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.rng = rng

        self.phi_length = self.network.num_frames
        self.image_width = self.network.input_width
        self.image_height = self.network.input_height

        # CREATE A FOLDER TO HOLD RESULTS
        time_str = time.strftime("_%m-%d-%H-%M_", time.gmtime())
        self.exp_dir = self.exp_pref + time_str + \
                       "{}".format(self.network.lr).replace(".", "p") + "_" \
                       + "{}".format(self.network.discount).replace(".", "p")

        try:
            os.stat(self.exp_dir)
        except OSError:
            os.makedirs(self.exp_dir)

        self.data_set = ale_data_set.DataSet(width=self.image_width,
                                             height=self.image_height,
                                             rng=rng,
                                             max_steps=self.replay_memory_size,
                                             phi_length=self.phi_length)

        # just needs to be big enough to create phi's
        self.test_data_set = ale_data_set.DataSet(width=self.image_width,
                                                  height=self.image_height,
                                                  rng=rng,
                                                  max_steps=self.phi_length * 2,
                                                  phi_length=self.phi_length)
        self.epsilon = self.epsilon_start
        if self.epsilon_decay != 0:
            self.epsilon_rate = ((self.epsilon_start - self.epsilon_min) /
                                 self.epsilon_decay)
        else:
            self.epsilon_rate = 0

        self.testing = False

        self._open_results_file()
        self._open_learning_file()

        self.step_counter = None
        self.episode_reward = None
        self.start_time = None
        self.loss_averages = None
        self.total_reward = None

        self.episode_counter = 0
        self.batch_counter = 0

        self.holdout_data = None

        # In order to add an element to the data set we need the
        # previous state and action and the current reward.  These
        # will be used to store states and actions.
        self.last_img = None
        self.last_action = None

        # Exponential moving average of runtime performance.
        self.steps_sec_ema = 0.
        self.program_start_time = None
        self.last_count_time = None
        self.epoch_time = None
        self.total_time = None

    def time_count_start(self):
        self.last_count_time = self.program_start_time = time.time()

    def _open_results_file(self):
        logging.info("OPENING " + self.exp_dir + '/results.csv')
        self.results_file = open(self.exp_dir + '/results.csv', 'w', 0)
        self.results_file.write(\
            'epoch,num_episodes,total_reward,reward_per_epoch,mean_q, epoch time, total time\n')
        self.results_file.flush()

    def _open_learning_file(self):
        self.learning_file = open(self.exp_dir + '/learning.csv', 'w', 0)
        self.learning_file.write('mean_loss,epsilon\n')
        self.learning_file.flush()

    def _update_results_file(self, epoch, num_episodes, holdout_sum):
        out = "{},{},{},{},{},{},{}\n".format(epoch, num_episodes, self.total_reward,
                                        self.total_reward / float(num_episodes),
                                        holdout_sum, self.epoch_time, self.total_time)
        self.last_count_time = time.time()
        self.results_file.write(out)
        self.results_file.flush()

    def _update_learning_file(self):
        out = "{},{}\n".format(np.mean(self.loss_averages),
                               self.epsilon)
        self.learning_file.write(out)
        self.learning_file.flush()

    def start_episode(self, observation):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - height x width numpy array

        Returns:
           An integer action
        """

        self.step_counter = 0
        self.batch_counter = 0
        self.episode_reward = 0

        # We report the mean loss for every epoch.
        self.loss_averages = []

        self.start_time = time.time()
        return_action = self.rng.randint(0, self.num_actions)

        self.last_action = return_action

        self.last_img = observation

        return return_action

    def _choose_action(self, data_set, epsilon, cur_img, reward):
        """
        Add the most recent data to the data set and choose
        an action based on the current policy.
        """

        data_set.add_sample(self.last_img, self.last_action, reward, False)
        if self.step_counter >= self.phi_length:
            phi = data_set.phi(cur_img)
            action = self.network.choose_action(phi, epsilon)
        else:
            action = self.rng.randint(0, self.num_actions)

        return action

    def _do_training(self):
        """
        Returns the average loss for the current batch.
        May be overridden if a subclass needs to train the network
        differently.
        """
        imgs, actions, rewards, terminals = self.data_set.random_batch(self.network.batch_size)
        evaluation = np.zeros((self.network.batch_size, 1), np.float32)
        for i in range(self.network.batch_size):
            state = imgs[i][self.data_set.phi_length-1]
            evaluation[i] = self.qec_table.estimate(state, actions[i])

        return self.network.train(imgs, actions, rewards, terminals, evaluation)

    def step(self, reward, observation):
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - A height x width numpy array

        Returns:
           An integer action.

        """

        self.step_counter += 1
        self.episode_reward += reward

        # TESTING---------------------------
        if self.testing:
            self.test_data_set.add_sample(self.last_img, self.last_action, np.clip(reward, -1, 1), False)
            if self.step_counter >= self.phi_length:
                phi = self.test_data_set.phi(observation)
                q_values = self.network.q_vals(phi)
                action1 = np.argmax(q_values)
                return1 = q_values[action1]
            else:
                action1 = 0
                return1 = -1

            action2 = 0
            return2 = 0
            for a in range(self.num_actions):
                return_a = self.qec_table.estimate(observation, a)
                if return_a > return2:
                    return2 = return_a
                    action2 = a
            print action1, return1, action2, return2
            raw_input()
            if return2 > return1*1.5:
                action = action2
            else:
                action = action1

        # NOT TESTING---------------------------
        else:
            if len(self.data_set) > self.replay_start_size:
                self.epsilon = max(self.epsilon_min,
                                   self.epsilon - self.epsilon_rate)

                action = self._choose_action(self.data_set, self.epsilon,
                                             observation,
                                             np.clip(reward, -1, 1))

                if self.step_counter % self.update_frequency == 0:
                    loss = self._do_training()
                    self.batch_counter += 1
                    self.loss_averages.append(loss)

            else:  # Still gathering initial random data...
                action = self._choose_action(self.data_set, self.epsilon,
                                             observation,
                                             np.clip(reward, -1, 1))

        self.last_action = action
        self.last_img = observation

        return action

    def end_episode(self, reward, terminal=True):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.
           terminal    - Whether the episode ended intrinsically
                         (ie we didn't run out of steps)
        Returns:
            None
        """

        self.episode_reward += reward
        self.step_counter += 1
        total_time = time.time() - self.start_time

        if self.testing:
            # If we run out of time, only count the last episode if
            # it was the only episode.
            if terminal or self.episode_counter == 0:
                self.episode_counter += 1
                self.total_reward += self.episode_reward
        else:
            # Store the latest sample.
            self.data_set.add_sample(self.last_img,
                                     self.last_action,
                                     np.clip(reward, -1, 1),
                                     True)
            """update"""
            q_return = 0.
            # last_q_return = -1.0
            index = (self.data_set.top-1) % self.data_set.max_steps
            while True:
                q_return = q_return * self.network.discount + self.data_set.rewards[index]
                # if not np.isclose(q_return, last_q_return):
                #     self.qec_table.update(self.data_set.imgs[index], self.data_set.actions[index], q_return)
                #     last_q_return = q_return
                self.qec_table.update(self.data_set.imgs[index], self.data_set.actions[index], q_return)
                index = (index-1) % self.data_set.max_steps
                if self.data_set.terminal[index] or index == self.data_set.bottom:
                    break

            rho = 0.98
            self.steps_sec_ema *= rho
            self.steps_sec_ema += (1. - rho) * (self.step_counter/total_time)

            logging.info("steps/second: {:.2f}, avg: {:.2f}".format(
                self.step_counter/total_time, self.steps_sec_ema))

            if self.batch_counter > 0:
                self._update_learning_file()
                logging.info("average loss: {:.4f}".format(\
                                np.mean(self.loss_averages)))

    def finish_epoch(self, epoch):
        # so large that i only keep one
        qec_file = open(self.exp_dir + '/qec_table_file_' + \
                        '.pkl', 'w')
        cPickle.dump(self.qec_table, qec_file, 2)
        qec_file.close()

        net_file = open(self.exp_dir + '/network_file_' + str(epoch) + \
                        '.pkl', 'w')
        cPickle.dump(self.network, net_file, -1)
        net_file.close()
        this_time = time.time()
        self.total_time = this_time-self.program_start_time
        self.epoch_time = this_time-self.last_count_time

    def start_testing(self):
        self.testing = True
        self.total_reward = 0
        self.episode_counter = 0

    def finish_testing(self, epoch):
        self.testing = False
        holdout_size = 3200

        if self.holdout_data is None and len(self.data_set) > holdout_size:
            imgs, _, _, _ = self.data_set.random_batch(holdout_size)
            self.holdout_data = imgs[:, :self.phi_length]

        holdout_sum = 0
        if self.holdout_data is not None:
            for i in range(holdout_size):
                holdout_sum += np.max(
                    self.network.q_vals(self.holdout_data[i]))

        self._update_results_file(epoch, self.episode_counter,
                                  holdout_sum / holdout_size)


class NeuralNetworkEpisodicMemory1(object):
    def __init__(self, q_network, qec_table, epsilon_start, epsilon_min,
                 epsilon_decay, replay_memory_size, exp_pref,
                 replay_start_size, update_frequency, ec_discount, num_actions, ec_testing, rng, KNN_decision=50000):
        self.network = q_network
        self.qec_table = qec_table
        self.ec_testing = ec_testing
        self.ec_discount = ec_discount
        self.num_actions = num_actions
        self.epsilon_start = epsilon_start

        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_memory_size = replay_memory_size
        self.exp_pref = exp_pref
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.rng = rng
        self.KNN_decision = KNN_decision

        self.phi_length = self.network.num_frames
        self.image_width = self.network.input_width
        self.image_height = self.network.input_height

        # CREATE A FOLDER TO HOLD RESULTS
        time_str = time.strftime("_%m-%d-%H-%M_", time.gmtime())
        self.exp_dir = self.exp_pref + time_str + \
                       "{}".format(self.network.lr).replace(".", "p") + "_" \
                       + "{}".format(self.network.discount).replace(".", "p")

        try:
            os.stat(self.exp_dir)
        except OSError:
            os.makedirs(self.exp_dir)

        self.data_set = ale_data_set.DataSet(width=self.image_width,
                                             height=self.image_height,
                                             rng=rng,
                                             max_steps=self.replay_memory_size,
                                             phi_length=self.phi_length)

        # just needs to be big enough to create phi's
        self.test_data_set = ale_data_set.DataSet(width=self.image_width,
                                                  height=self.image_height,
                                                  rng=rng,
                                                  max_steps=self.phi_length * 2,
                                                  phi_length=self.phi_length)
        self.epsilon = self.epsilon_start
        if self.epsilon_decay != 0:
            self.epsilon_rate = ((self.epsilon_start - self.epsilon_min) /
                                 self.epsilon_decay)
        else:
            self.epsilon_rate = 0

        self.testing = False

        self._open_results_file()
        self._open_learning_file()

        self.step_counter = None
        self.episode_reward = None
        self.start_time = None
        self.loss_averages = None
        self.total_reward = None

        self.episode_counter = 0
        self.batch_counter = 0

        self.holdout_data = None

        # In order to add an element to the data set we need the
        # previous state and action and the current reward.  These
        # will be used to store states and actions.
        self.last_img = None
        self.last_action = None

        # Exponential moving average of runtime performance.
        self.steps_sec_ema = 0.
        self.program_start_time = None
        self.last_count_time = None
        self.epoch_time = None
        self.total_time = None

    def time_count_start(self):
        self.last_count_time = self.program_start_time = time.time()

    def _open_results_file(self):
        logging.info("OPENING " + self.exp_dir + '/results.csv')
        self.results_file = open(self.exp_dir + '/results.csv', 'w', 0)
        self.results_file.write(\
            'epoch,num_episodes,total_reward,reward_per_epoch,mean_q, epoch time, total time\n')
        self.results_file.flush()

    def _open_learning_file(self):
        self.learning_file = open(self.exp_dir + '/learning.csv', 'w', 0)
        self.learning_file.write('mean_loss,epsilon\n')
        self.learning_file.flush()

    def _update_results_file(self, epoch, num_episodes, holdout_sum):
        out = "{},{},{},{},{},{},{}\n".format(epoch, num_episodes, self.total_reward,
                                        self.total_reward / float(num_episodes),
                                        holdout_sum, self.epoch_time, self.total_time)
        self.last_count_time = time.time()
        self.results_file.write(out)
        self.results_file.flush()

    def _update_learning_file(self):
        out = "{},{}\n".format(np.mean(self.loss_averages),
                               self.epsilon)
        self.learning_file.write(out)
        self.learning_file.flush()

    def start_episode(self, observation):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - height x width numpy array

        Returns:
           An integer action
        """

        self.step_counter = 0
        self.batch_counter = 0
        self.episode_reward = 0

        # We report the mean loss for every epoch.
        self.loss_averages = []

        self.start_time = time.time()
        return_action = self.rng.randint(0, self.num_actions)

        self.last_action = return_action

        self.last_img = observation

        return return_action

    def _choose_action(self, data_set, epsilon, cur_img, reward):
        """
        Add the most recent data to the data set and choose
        an action based on the current policy.
        """

        data_set.add_sample(self.last_img, self.last_action, reward, False)
        if self.step_counter >= self.phi_length:
            phi = data_set.phi(cur_img)

            # if len(self.data_set) < self.KNN_decision:
            #     if self.rng.rand() < epsilon:
            #         return self.rng.randint(0, self.num_actions)
            #     value = -100
            #     maximum_action = 0
            #     for action in range(self.num_actions):
            #         value_t = self.qec_table.estimate(cur_img, action)
            #         if value_t > value:
            #             value = value_t
            #             maximum_action = action
            #     return maximum_action

            action = self.network.choose_action(phi, epsilon)
        else:
            action = self.rng.randint(0, self.num_actions)

        return action

    def _do_training(self):
        """
        Returns the average loss for the current batch.
        May be overridden if a subclass needs to train the network
        differently.
        """
        imgs, actions, rewards, terminals = self.data_set.random_batch(self.network.batch_size)
        evaluation = np.zeros((self.network.batch_size, 1), np.float32)
        for i in range(self.network.batch_size):
            state = imgs[i][self.data_set.phi_length-1]
            evaluation[i] = self.qec_table.estimate(state, actions[i])
            print rewards[i], evaluation[i]
            evaluation[i] = np.maximum(rewards[i], evaluation[i])

        return self.network.train(imgs, actions, rewards, terminals, evaluation)

    def step(self, reward, observation):
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - A height x width numpy array

        Returns:
           An integer action.

        """

        self.step_counter += 1
        self.episode_reward += reward

        # TESTING---------------------------
        if self.testing:
            action = self._choose_action(self.test_data_set, 0.0,
                                         observation, np.clip(reward, -1, 1))

        # NOT TESTING---------------------------
        else:
            if len(self.data_set) > self.replay_start_size:
                self.epsilon = max(self.epsilon_min,
                                   self.epsilon - self.epsilon_rate)

                action = self._choose_action(self.data_set, self.epsilon,
                                             observation,
                                             np.clip(reward, -1, 1))

                if self.step_counter % self.update_frequency == 0:
                    loss = self._do_training()
                    self.batch_counter += 1
                    self.loss_averages.append(loss)

            else:  # Still gathering initial random data...
                action = self._choose_action(self.data_set, self.epsilon,
                                             observation,
                                             np.clip(reward, -1, 1))

        self.last_action = action
        self.last_img = observation

        return action

    def end_episode(self, reward, terminal=True):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.
           terminal    - Whether the episode ended intrinsically
                         (ie we didn't run out of steps)
        Returns:
            None
        """

        self.episode_reward += reward
        self.step_counter += 1
        total_time = time.time() - self.start_time

        if self.testing:
            # If we run out of time, only count the last episode if
            # it was the only episode.
            if terminal or self.episode_counter == 0:
                self.episode_counter += 1
                self.total_reward += self.episode_reward
        else:
            # Store the latest sample.
            self.data_set.add_sample(self.last_img,
                                     self.last_action,
                                     np.clip(reward, -1, 1),
                                     True)
            """update"""
            q_return = 0.
            # last_q_return = -1.0
            index = (self.data_set.top-1) % self.data_set.max_steps
            while True:
                q_return = q_return * self.network.discount + self.data_set.rewards[index]
                # if not np.isclose(q_return, last_q_return):
                #     self.qec_table.update(self.data_set.imgs[index], self.data_set.actions[index], q_return)
                #     last_q_return = q_return
                self.qec_table.update(self.data_set.imgs[index], self.data_set.actions[index], q_return)
                index = (index-1) % self.data_set.max_steps
                if self.data_set.terminal[index] or index == self.data_set.bottom:
                    break

            rho = 0.98
            self.steps_sec_ema *= rho
            self.steps_sec_ema += (1. - rho) * (self.step_counter/total_time)

            logging.info("steps/second: {:.2f}, avg: {:.2f}".format(
                self.step_counter/total_time, self.steps_sec_ema))

            if self.batch_counter > 0:
                self._update_learning_file()
                logging.info("average loss: {:.4f}".format(\
                                np.mean(self.loss_averages)))

    def finish_epoch(self, epoch):
        # so large that i only keep one
        qec_file = open(self.exp_dir + '/qec_table_file_' + \
                        '.pkl', 'w')
        cPickle.dump(self.qec_table, qec_file, 2)
        qec_file.close()

        net_file = open(self.exp_dir + '/network_file_' + str(epoch) + \
                        '.pkl', 'w')
        cPickle.dump(self.network, net_file, -1)
        net_file.close()
        this_time = time.time()
        self.total_time = this_time-self.program_start_time
        self.epoch_time = this_time-self.last_count_time

    def start_testing(self):
        self.testing = True
        self.total_reward = 0
        self.episode_counter = 0

    def finish_testing(self, epoch):
        self.testing = False
        holdout_size = 3200

        if self.holdout_data is None and len(self.data_set) > holdout_size:
            imgs, _, _, _ = self.data_set.random_batch(holdout_size)
            self.holdout_data = imgs[:, :self.phi_length]

        holdout_sum = 0
        if self.holdout_data is not None:
            for i in range(holdout_size):
                holdout_sum += np.max(
                    self.network.q_vals(self.holdout_data[i]))

        self._update_results_file(epoch, self.episode_counter,
                                  holdout_sum / holdout_size)
