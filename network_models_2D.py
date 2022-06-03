import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from loss_functions import huber_loss
from network_2D import FCQN

class initialize_network_FCQN:
    def __init__(self, plot_directory: str, save_weights_dir: str, custom_load_path=None):
        self.g = tf.Graph()

        with self.g.as_default():
            stat_writer_path = plot_directory
            self.save_weights_path = save_weights_dir
            self.stat_writer = tf.summary.FileWriter(stat_writer_path)

            self.input_size = 224

            # Placeholders
            self.batch_size = tf.placeholder(tf.int32, shape=())
            self.learning_rate = tf.placeholder(tf.float32, shape=())

            self.sampled_grids = tf.placeholder(tf.float32, [None, 224, 224, 1], name='BHM_Linearly_Sampled_Grids')
            self.current_position_map1 = tf.placeholder(tf.float32, [None, 224, 224, 1], name='Current_pos_map1')
            self.current_position_map2 = tf.placeholder(tf.float32, [None, 224, 224, 1], name='Current_pos_map2')
            self.previous_position_map1 = tf.placeholder(tf.float32, [None, 224, 224, 1], name='Previous_pos_map1')
            self.previous_position_map2 = tf.placeholder(tf.float32, [None, 224, 224, 1], name='Previous_pos_map2')
            self.X = tf.concat((self.sampled_grids, self.current_position_map1, self.previous_position_map1, self.current_position_map2, self.previous_position_map2), axis=3, name='Combined_image')

            # target will be fed later as Qvals of next state (which is Reward gained + discount * max(Q of next s a) - Q of current s a)
            self.target = tf.placeholder(tf.float32, shape=[None],      name='Target_Qs')           
            self.action = tf.placeholder(tf.int32, shape=[None, 2],   name='Actions')         
            self.model = FCQN(self.X)
            self.action_space = 52 ** 2
            self.predict = self.model.output   # for simplicity, i flattened my model output to 2704 (52 ** 2)
            action_flattened_index = self.action[0][0] * 52 + self.action[0][1]
            # self.actions = np.ravel_multi_index(self.actions, self.model.output.shape)
            # print('action:', self.action[0])
            # print('action flattened index:', action_flattened_index.shape)
            ind = tf.one_hot(action_flattened_index, self.action_space)            

            # self.flattened_output = tf.placeholder(tf.float32, shape=(None, 2704), name='Flattened_weights')

            # flattened_output = tf.reshape(self.model.output, [-1])
            # print('flattened_output', self.flattened_output)
            pred_Q = tf.reduce_sum(
                tf.multiply(self.model.output, ind),
                axis=0)

            # print('pred_Q1', pred_Q1)
            # print('pred_Q2', pred_Q2)
            self.loss = huber_loss(pred_Q, self.target)     # original paper used MSE           
            self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99).minimize(
                self.loss, name="train")
            self.sess = tf.InteractiveSession()
            self.saver = tf.train.Saver()
            self.all_vars = tf.trainable_variables()
            self.sess.graph.finalize()
        # Load custom weights from custom_load_path if required
        if custom_load_path is not None:
            try:
                print('Loading weights from: ', custom_load_path)
                self.load_network(custom_load_path)
            except:
                pass

    def get_vars(self):
        return self.sess.run(self.all_vars)

    def Q_val(self, curr_state_tuple1, curr_state_tuple2):
        grids, curr_positions1, prev_positions1 = curr_state_tuple1
        grids, curr_positions2, prev_positions2 = curr_state_tuple2
        print("HELLO")
        return self.sess.run(self.predict,
                             feed_dict={self.batch_size: grids.shape[0], self.sampled_grids: grids,
                                        self.current_position_map1: curr_positions1, self.previous_position_map1: prev_positions1,
                                        self.current_position_map2: curr_positions2, self.previous_position_map2: prev_positions2})
    
    def train_n(self, curr_state_tuple1, curr_state_tuple2, action, target_Q, batch_size, lr, epsilon, iter):
        grids, curr_positions1, prev_positions1 = curr_state_tuple1
        grids, curr_positions2, prev_positions2 = curr_state_tuple2
        _, loss, Q = self.sess.run([self.train, self.loss, self.predict],
                                   feed_dict={self.batch_size: batch_size, self.learning_rate: lr, self.sampled_grids: grids,
                                              self.current_position_map1: curr_positions1, self.previous_position_map1: prev_positions1,
                                              self.current_position_map2: curr_positions2, self.previous_position_map2: prev_positions2,
                                              self.action: action, self.target: target_Q})

        meanQ = np.mean(Q)
        maxQ = np.max(Q)
        # Log to tensorboard
        self.log_to_tensorboard(tag='Loss', group='drone_2D', value=np.linalg.norm(loss) / batch_size, index=iter)
        self.log_to_tensorboard(tag='Epsilon', group='drone_2D', value=epsilon, index=iter)
        # self.log_to_tensorboard(tag='Learning Rate', group='drone_2D', value=lr, index=iter)
        self.log_to_tensorboard(tag='MeanQ', group='drone_2D', value=meanQ, index=iter)
        self.log_to_tensorboard(tag='MaxQ', group='drone_2D', value=maxQ, index=iter)

    def action_selection(self, curr_state_tuple1, curr_state_tuple2):
        grids, curr_positions1, prev_positions1 = curr_state_tuple1
        grids, curr_positions2, prev_positions2 = curr_state_tuple2     
        qvals = self.sess.run(self.predict,
                              feed_dict={self.batch_size: grids.shape[0],
                                         self.sampled_grids: grids, self.current_position_map1: curr_positions1, self.current_position_map2: curr_positions2,
                                         self.previous_position_map1: prev_positions1, self.previous_position_map2: prev_positions2})

        # np.argmax gives me index into the flattened qvals array. np.unravel_index returns me the actual indexes of the original shape
        # print('model max action:', np.argmax(qvals))

        action = np.unravel_index(np.argmax(qvals), (52, 52))
        # print('converted action:', action)
        return np.array([action])

    def action_selection_non_repeat(self, curr_state_tuple1, curr_state_tuple2, previous_actions):
        """Same as action selection, but extra filtering to prevent repeated actions"""
        grids, curr_positions1, prev_positions1 = curr_state_tuple1
        grids, curr_positions2, prev_positions2 = curr_state_tuple2    
        qvals = self.sess.run(self.predict,
                              feed_dict={self.batch_size: grids.shape[0],
                                         self.sampled_grids: grids, self.current_position_map1: curr_positions1, self.current_position_map2: curr_positions2,
                                         self.previous_position_map1: prev_positions1, self.previous_position_map2: prev_positions2})
        while True:
            action_idx = np.unravel_index(np.argmax(qvals), (52, 52))
            repeated_action = action_idx in previous_actions

            if repeated_action:
                qvals[np.argmax(qvals)] = -np.inf
            else:
                return np.array([action_idx])

    def log_to_tensorboard(self, tag, group, value, index):
        summary = tf.Summary()
        tag = group + '/' + tag
        summary.value.add(tag=tag, simple_value=value)
        self.stat_writer.add_summary(summary, index)

    def save_network(self, iter=''):
        save_path = self.save_weights_path + '/drone_2D' + '_' + iter
        self.saver.save(self.sess, save_path)
        print('Model Saved: ', save_path)

    def load_network(self, load_path):
        self.saver.restore(self.sess, load_path)
