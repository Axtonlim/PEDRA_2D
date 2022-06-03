import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class FCQN:
    def __init__(self, X):    # TODO: NOW, HARDCODED AS 52x52 action space, can it be more flexible?
        self.X = X
        # print(self.X.shape)
        self.conv1 = self.conv(self.X, k=5, out=96, s=1, p='VALID')
        # print("SELF con1 shape", self.conv1.shape)
        self.maxpool1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
        # print("SELF maxpool1 shape", self.maxpool1.shape)

        self.conv2 = self.conv(self.maxpool1, k=5, out=64, s=1, p="VALID")
        # print("SELF con2 shape", self.conv2.shape)
        self.maxpool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        # print("SELF maxpool1 shape", self.maxpool2.shape)

        self.conv3 = self.conv(self.maxpool2, k=3, out=64, s=1, p="SAME")
        # print("SELF con3 shape", self.conv3.shape)

        # Advantage Network
        self.advantage = self.conv(self.conv3, k=1, out=1, s=1, p='VALID')
        # print("SELF advantage shape", self.advantage.shape)

        # Value Network (Global max pooling)
        self.value = tf.nn.max_pool(self.conv3, ksize=[1, 52, 52, 1], strides=[1, 1, 1, 1], padding='VALID')        # HARDCODED GLOBAL MAX POOL SIZE
        # print("SELF value shape", self.value.shape)

        # Q value of each point action
        self.output = tf.reshape((self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, keepdims=True)))[0, :, :, 0], [-1])  # flattened
        

    @staticmethod
    def conv(input, k, out, s, p, trainable=True):
        W = tf.Variable(tf.random.truncated_normal(shape=(k, k, int(input.shape[3]), out), stddev=0.05), trainable=trainable)
        b = tf.Variable(tf.random.truncated_normal(shape=[out], stddev=0.05), trainable=trainable)

        conv_kernel_1 = tf.nn.conv2d(input, W, [1, s, s, 1], padding=p)
        bias_layer_1 = tf.nn.bias_add(conv_kernel_1, tf.Variable(b, trainable))

        return tf.nn.relu(bias_layer_1)
