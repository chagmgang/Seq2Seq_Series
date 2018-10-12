import tensorflow as tf

class s2s:
    def __init__(self, enc_sent_size, output_sent_size, vocab_size):

        self.enc_input_size = enc_sent_size - 1

        self.enc_input = tf.placeholder(tf.float32, [None, self.enc_input_size, vocab_size], name='inputs')
        self.dec_input = tf.placeholder(tf.float32, [None, output_sent_size, vocab_size], name='outputs')
        # [batch size, time steps]
        self.targets = tf.placeholder(tf.int64, [None, None], name='targets')


        with tf.variable_scope('encode'):
            enc_cell = [tf.nn.rnn_cell.GRUCell(size) for size in [1024, 512, 256, self.enc_input_size]]
            enc_cell = tf.nn.rnn_cell.MultiRNNCell(enc_cell)
            outputs_enc, enc_states = tf.nn.dynamic_rnn(cell=enc_cell, inputs=self.enc_input, dtype=tf.float32)

        with tf.variable_scope('decode'):
            dec_cell = [tf.nn.rnn_cell.GRUCell(size) for size in [1024, 512, 256, self.enc_input_size]]
            dec_cell = tf.nn.rnn_cell.MultiRNNCell(dec_cell)
            outputs_dec, dec_states = tf.nn.dynamic_rnn(cell=dec_cell, inputs=self.dec_input, initial_state=enc_states,
                                        dtype=tf.float32)

        expand_outputs_dec = tf.expand_dims(outputs_dec, 2)
        expand_outputs_enc = tf.expand_dims(outputs_enc, 1)
        tile_outputs_dec = tf.tile(expand_outputs_dec, [1, 1, self.enc_input_size, 1])
        context_vector = tf.multiply(tile_outputs_dec, expand_outputs_enc)
        context_vector_reshape = tf.reshape(context_vector, [-1, output_sent_size, self.enc_input_size * self.enc_input_size])
        context_vector = tf.layers.dense(inputs=context_vector_reshape, units=self.enc_input_size, activation=None)
        self.context_vector = tf.nn.softmax(context_vector)

        output = tf.multiply(context_vector, outputs_dec)

        self.model = tf.layers.dense(output, vocab_size, activation=None)

        self.cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.model, labels=self.targets))

        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

        self.prediction = tf.argmax(self.model, 2, name='prediction')

'''
import tensorflow as tf
import numpy as np
from konlpy.tag import Twitter

twitter = Twitter()
input_sent = []
with open('/Users/chageumgang/Desktop/Seq2Seq_Series/input.log', 'r', encoding='utf-8') as content_file:
    for line in content_file:
        tag = twitter.pos(line)[:-1]
        input_sent.append([i[0] for i in tag])

output_sent = []
with open('/Users/chageumgang/Desktop/Seq2Seq_Series/output.log', 'r', encoding='utf-8') as content_file:
    for line in content_file:
        tag = twitter.pos(line)[:-1]
        output_sent.append([i[0] for i in tag])

vocab_list = []
with open('/Users/chageumgang/Desktop/Seq2Seq_Series/vocab.log', 'r', encoding='utf-8') as content_file:
    for line in content_file:
        vocab_list.append(line[:-1])

vocab_dict = {n: i for i, n in enumerate(vocab_list)}
num_dic = len(vocab_dict)


input_length = [len(i) for i in input_sent]
output_length = [len(o) for o in output_sent]
max_len_i = max(input_length)
max_len_o = max(output_length)

input_batch = []
output_batch = []
target_batch = []

for i, o in zip(input_sent, output_sent):
    while not len(i) == max_len_i:
        if len(i) < max_len_i:
            i.append('<%>')
    while not len(o) == max_len_o:
        if len(o) < max_len_o:
            o.append('<%>')
    real_i = i
    real_o = ['<start>'] + [x for x in o]
    real_target = [x for x in o] + ['<end>']

    input = [vocab_dict[n] for n in real_i]
    output = [vocab_dict[n] for n in real_o]
    target = [vocab_dict[n] for n in real_target]

    input_batch.append(np.eye(num_dic)[input])
    output_batch.append(np.eye(num_dic)[output])
    target_batch.append(target)


total_epoch = 3000
n_class = n_input = num_dic

enc_sent_size = max_len_i + 1
output_sent_size = max_len_o + 1
vocab_size = num_dic

S2S = s2s(enc_sent_size, output_sent_size, vocab_size)
'''