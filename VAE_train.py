import os
import pickle
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.python.layers import core as core_layers
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
import time
# import jieba
from Util import mybleu
from Util import myResidualCell
import random
import pickle as cPickle
import matplotlib.pyplot as plt
import nltk

w2id, id2w = pickle.load(open('shakespearian-corpus/w2id_id2w.pkl','rb'))


def idx2str(s):
    return ' '.join([id2w[idx] for idx in s])

def str2idx(idx):
    idx = idx.strip()
    return [w2id[idxx] for idxx in idx.split()]

def pad(x, pid, move_go=False):
    max_length = 30
    x = [k[:max_length] for k in x]
    if move_go:
        length_list = [len(k) - 1 for k in x]
    else:
        length_list = [len(k) for k in x]
    max_length = max(length_list)
    pad_x = []
    for k in x:
        if move_go:
            pad_k = k[1:] + [pid, ] * (max_length - len(k[1:]))
        else:
            pad_k = k + [pid, ] * (max_length - len(k))
        pad_x.append(pad_k)
    return pad_x, length_list

def pad_maxlength(x, pid, move_go=False):
    max_length = 30
    if move_go:
        length_list = [len(k) - 1 for k in x]
    else:
        length_list = [min(len(k), max_length) for k in x]

    pad_x = []
    for k in x:
        if move_go:
            pad_k = k[1:] + [pid, ] * (max_length - len(k[1:]))
        else:
            pad_k = k[:max_length] + [pid, ] * (max_length - len(k))
        pad_x.append(pad_k)
    return pad_x, length_list


# setting session config
tf.logging.set_verbosity(tf.logging.INFO)
sess_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))


def word_overlap_edit(s1, s2):
    t1 = set(s1.split())
    t2 = set(s2.split())
    word_overlap = float(len(t1 & t2)) / len(t1 | t2)
    edit_distance = 1 - float(nltk.edit_distance(s1.split(), s2.split())) /  max(len(s1.split()), len(s2.split()))
    return word_overlap, edit_distance


class VAE:
    def __init__(self, dp, rnn_size, n_layers, Lambda, gamma, num_classes, latent_dim, encoder_embedding_dim,
                 decoder_embedding_dim, max_infer_length,
                 sess, lr=0.001, grad_clip=5.0, beam_width=10, force_teaching_ratio=1.0, beam_penalty=1.0,
                 residual=False, output_keep_prob=0.5, input_keep_prob=0.9, bow_size=400, predictor_size=None,
                 is_inference=False, latent_weight=0.4, beta_decay_period=10, beta_decay_offset=5, cell_type='lstm',
                 reverse=False,
                 decay_scheme='luong234'):

        self.rnn_size = rnn_size
        self.n_layers = n_layers
        self.bow_size = bow_size
        if not predictor_size:
            self.predictor_size = self.rnn_size * 4
        else:
            self.predictor_size = predictor_size
        self.Lambda = Lambda
        self.grad_clip = grad_clip
        self.dp = dp
        self.latent_weight = latent_weight
        self.beta_decay_period = beta_decay_period
        self.beta_decay_offset = beta_decay_offset
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.step = 0
        self.num_classes = num_classes
        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_embedding_dim = decoder_embedding_dim
        self.beam_width = beam_width
        self.is_inference = is_inference
        self.beam_penalty = beam_penalty
        self.max_infer_length = max_infer_length
        self.residual = residual
        self.decay_scheme = decay_scheme
        if self.residual:
            assert encoder_embedding_dim == rnn_size
            assert decoder_embedding_dim == rnn_size
        self.reverse = reverse
        self.cell_type = cell_type
        self.force_teaching_ratio = force_teaching_ratio
        self._output_keep_prob = output_keep_prob
        self._input_keep_prob = input_keep_prob
        self.sess = sess
        self.lr = lr
        self.build_graph()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=35)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()

    # end constructor

    def build_graph(self):
        self.register_symbols()
        self.add_input_layer()
        self.add_encoder_layer()
        self.add_stochastic_layer()
        self.add_decoder_hidden()
        with tf.variable_scope('decode'):
            self.add_decoder_for_training()
        with tf.variable_scope('decode', reuse=True):
            self.add_decoder_for_inference()
        with tf.variable_scope('decode', reuse=True):
            self.add_decoder_for_prefix_inference()
        with tf.variable_scope('predictor_layer'):
            self.add_classifer()
        self.add_backward_path()

    # end method

    def _item_or_tuple(self, seq):
        """Returns `seq` as tuple or the singular element.
        Which is returned is determined by how the AttentionMechanism(s) were passed
        to the constructor.
        Args:
          seq: A non-empty sequence of items or generator.
        Returns:
           Either the values in the sequence as a tuple if AttentionMechanism(s)
           were passed to the constructor as a sequence or the singular element.
        """
        t = tuple(seq)
        if self._is_multi:
            return t
        else:
            return t[0]

    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, None], name="X")
        self.Y = tf.placeholder(tf.int32, [None, None], name="Y")
        self.X_seq_len = tf.placeholder(tf.int32, [None], name="X_seq_len")
        self.Y_seq_len = tf.placeholder(tf.int32, [None], name="Y_seq_len")
        self.C = tf.placeholder(tf.int32, [None, self.num_classes], name='C')
        self.input_keep_prob = tf.placeholder(tf.float32, name="input_keep_prob")
        self.output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")
        self.batch_size = tf.shape(self.X)[0]
        self.B = tf.placeholder(tf.float32, name='Beta_deterministic_warmup')
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.predictor_global_step = tf.Variable(0, name="predictor_global_step", trainable=False)
        self.bow_global_step = tf.Variable(0, name="bow_global_step", trainable=False)

    # end method

    def single_cell(self, reuse=False):
        if self.cell_type == 'lstm':
            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.rnn_size, reuse=reuse)
        else:
            cell = tf.contrib.rnn.GRUBlockCell(self.rnn_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell, self.output_keep_prob, self.input_keep_prob)
        if self.residual:
            cell = myResidualCell.ResidualWrapper(cell)
        return cell

    def add_encoder_layer(self):
        encoder_embedding = tf.get_variable('encoder_embedding', [len(self.dp.X_w2id), self.encoder_embedding_dim],
                                            tf.float32, tf.random_uniform_initializer(-1.0, 1.0))

        self.encoder_inputs = tf.nn.embedding_lookup(encoder_embedding, self.X)
        bi_encoder_output, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=tf.contrib.rnn.MultiRNNCell([self.single_cell() for _ in range(self.n_layers)]),
            cell_bw=tf.contrib.rnn.MultiRNNCell([self.single_cell() for _ in range(self.n_layers)]),
            inputs=self.encoder_inputs,
            sequence_length=self.X_seq_len,
            dtype=tf.float32,
            scope='bidirectional_rnn')

        if self.cell_type == 'lstm':
            self.encoder_out = tf.concat([bi_encoder_state[0][-1][1], bi_encoder_state[1][-1][1]], -1)
        else:
            self.encoder_out = tf.concat([bi_encoder_state[0][-1], bi_encoder_state[1][-1]], -1)

        # print('encoder_out', self.encoder_out)

    def add_stochastic_layer(self):
        # self.z_mu = tf.layers.dense(self.encoder_out, self.latent_dim)
        # self.z_lgs2 = tf.layers.dense(self.encoder_out, self.latent_dim)
        # noise = tf.random_normal(tf.shape(self.z_lgs2))
        self.z_mu = tf.layers.dense(self.encoder_out, self.latent_dim)
        self.z_lgs2 = tf.layers.dense(self.encoder_out, self.latent_dim)
        noise = tf.random_normal(tf.shape(self.z_lgs2))
        if self.is_inference:
            self.z = self.z_mu
        else:
            self.z = self.z_mu + tf.exp(0.5 * self.z_lgs2) * noise
        with tf.variable_scope('bow_layer'):
            self.bow_fc1 = tf.layers.dense(self.z, self.bow_size, activation=tf.tanh, name="bow_fc1")
            self.bow_fc1 = tf.nn.dropout(self.bow_fc1, self.output_keep_prob)
            # print('bow_fc1', self.bow_fc1)
            self.bow_logits = tf.layers.dense(self.bow_fc1, len(self.dp.Y_w2id), activation=None, name="bow_project")
            # print('bow_logits', self.bow_logits)

    def add_classifer(self):
        # print('self.encoder_out', self.encoder_out)
        h_hat = self.z
        # self.z = self.encoder_out
        fc = tf.layers.dense(h_hat, self.predictor_size, name='fc1')
        fc = tf.contrib.layers.dropout(fc, self.output_keep_prob)
        fc = tf.nn.relu(fc)
        self.fc = fc
        self.logits = tf.layers.dense(fc, self.num_classes, name='fc2')
        # self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
        self.ypred_for_auc = tf.nn.softmax(self.logits)
        self.predictions = tf.argmax(self.logits, 1, name="predictions")

        correct_pred = tf.equal(self.predictions, tf.argmax(self.C, 1))
        # print('correct_pred', correct_pred)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def add_decoder_hidden(self):
        hidden_state_list = []
        for i in range(self.n_layers * 2):
            if self.cell_type == 'gru':
                hidden_state_list.append(tf.layers.dense(self.z, self.rnn_size))
            else:
                hidden_state_list.append(tf.contrib.rnn.LSTMStateTuple(tf.layers.dense(self.z, self.rnn_size),
                                                                       tf.layers.dense(self.z, self.rnn_size)))
        self.decoder_init_state = tuple(hidden_state_list)
        # print('self.decoder_init_state', self.decoder_init_state)

    def processed_decoder_input(self):
        main = tf.strided_slice(self.Y, [0, 0], [self.batch_size, -1], [1, 1])  # remove last char
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], self._y_go), main], 1)
        return decoder_input

    def add_decoder_for_training(self):
        self.decoder_cell = tf.contrib.rnn.MultiRNNCell([self.single_cell() for _ in range(2 * self.n_layers)])
        decoder_embedding = tf.get_variable('decoder_embedding', [len(self.dp.Y_w2id), self.decoder_embedding_dim],
                                            tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
        emb = tf.nn.embedding_lookup(decoder_embedding, self.processed_decoder_input())
        inputs = tf.expand_dims(self.z, 1)
        inputs = tf.tile(inputs, [1, tf.shape(emb)[1], 1])
        inputs = tf.concat([emb, inputs], 2)
        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=inputs,
            sequence_length=self.Y_seq_len,
            time_major=False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            helper=training_helper,
            initial_state=self.decoder_init_state,  # self.decoder_cell.zero_state(self.batch_size, tf.float32),
            output_layer=core_layers.Dense(len(self.dp.Y_w2id)))
        training_decoder_output, training_final_state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=training_decoder,
            impute_finished=True,
            maximum_iterations=tf.reduce_max(self.Y_seq_len))
        self.training_logits = training_decoder_output.rnn_output
        self.init_prefix_state = training_final_state

    def add_decoder_for_inference(self):
        decoder_embedding = tf.get_variable('decoder_embedding')
        self.beam_f = (lambda ids: tf.concat([tf.nn.embedding_lookup(decoder_embedding, ids),
                                              tf.tile(tf.expand_dims(self.z, 1),
                                                      [1, int(
                                                          tf.nn.embedding_lookup(decoder_embedding, ids).get_shape()[
                                                              1]), 1]) if len(ids.get_shape()) != 1
                                              else self.z], -1))

        predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=self.decoder_cell,
            embedding=self.beam_f,
            start_tokens=tf.tile(tf.constant([self._y_go], dtype=tf.int32), [self.batch_size]),
            end_token=self._y_eos,
            initial_state=tf.contrib.seq2seq.tile_batch(self.decoder_init_state, self.beam_width),
            # self.decoder_cell.zero_state(self.batch_size * self.beam_width, tf.float32),
            beam_width=self.beam_width,
            output_layer=core_layers.Dense(len(self.dp.Y_w2id), _reuse=True),
            length_penalty_weight=self.beam_penalty)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=predicting_decoder,
            impute_finished=False,
            maximum_iterations=self.max_infer_length)
        self.predicting_ids = predicting_decoder_output.predicted_ids
        self.score = predicting_decoder_output.beam_search_decoder_output.scores

    def add_decoder_for_prefix_inference(self):
        predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=self.decoder_cell,
            embedding=self.beam_f,
            start_tokens=tf.tile(tf.constant([self._y_go], dtype=tf.int32), [self.batch_size]),
            end_token=self._y_eos,
            initial_state=tf.contrib.seq2seq.tile_batch(self.init_prefix_state, self.beam_width),
            beam_width=self.beam_width,
            output_layer=core_layers.Dense(len(self.dp.Y_w2id), _reuse=True),
            length_penalty_weight=self.beam_penalty)

        self.prefix_go = tf.placeholder(tf.int32, [None])
        prefix_go_beam = tf.tile(tf.expand_dims(self.prefix_go, 1), [1, self.beam_width])
        prefix_emb = self.beam_f(prefix_go_beam)
        predicting_decoder._start_inputs = prefix_emb
        predicting_prefix_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=predicting_decoder,
            impute_finished=False,
            maximum_iterations=self.max_infer_length)
        self.predicting_prefix_ids = predicting_prefix_decoder_output.predicted_ids
        self.prefix_score = predicting_prefix_decoder_output.beam_search_decoder_output.scores

    def add_backward_path(self):
        # print(self.logits, self.C)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.C)
        self.c_loss = tf.reduce_mean(cross_entropy)

        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.r_loss = tf.contrib.seq2seq.sequence_loss(logits=self.training_logits,
                                                       targets=self.Y,
                                                       weights=masks)
        self.all_reconstruct_loss = tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(logits=self.training_logits,
                                                                                   targets=self.Y,
                                                                                   weights=masks,
                                                                                   average_across_timesteps=False))
        self.kl_loss = tf.reduce_mean(
            -0.5 * tf.reduce_sum(1 + self.z_lgs2 - tf.square(self.z_mu) - tf.exp(self.z_lgs2), 1))

        max_out_len = array_ops.shape(self.Y)[1]
        self.tile_bow_logits = tf.tile(tf.expand_dims(self.bow_logits, 1), [1, max_out_len, 1])
        labels = self.Y
        label_mask = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        bow_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.tile_bow_logits,
                                                                  labels=labels) * label_mask
        bow_loss = tf.reduce_sum(bow_loss, reduction_indices=1)
        self.avg_bow_loss = tf.reduce_mean(bow_loss)

        self.loss = self.Lambda * self.c_loss + (1 - self.Lambda) * (
                    self.r_loss + self.B * self.latent_weight * self.kl_loss + self.avg_bow_loss)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.learning_rate = tf.constant(self.lr)
        self.learning_rate = self.get_learning_rate_decay(self.decay_scheme)  # decay
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(clipped_gradients, params),
                                                                                   global_step=self.global_step)

        self.Dgrad = tf.gradients(self.c_loss, [self.z])
        self.Dgradmu = tf.gradients(self.c_loss, [self.z_mu])

        # ---- predictor -----#
        params_predictor = [v for v in tf.trainable_variables() if 'predictor_layer' in v.name]
        print('params_predictor', params_predictor)
        gradients_predictor = tf.gradients(self.c_loss, params_predictor)
        # clipped_gradients_predictor, _ = tf.clip_by_global_norm(gradients_predictor, self.grad_clip)
        self.predictor_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
            zip(gradients_predictor, params_predictor), global_step=self.predictor_global_step)

        # ---- bow---------#
        params_bow = [v for v in tf.trainable_variables() if 'bow_layer' in v.name]
        print('params_bow', params_bow)
        gradients_bow = tf.gradients(self.avg_bow_loss, params_bow)
        # clipped_gradients_bow, _ = tf.clip_by_global_norm(gradients_bow, self.grad_clip)
        self.bow_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(gradients_bow, params_bow),
                                                                                 global_step=self.bow_global_step)

        self.Bowgradmu = tf.gradients(self.avg_bow_loss, [self.z_mu])

    def register_symbols(self):
        self._x_go = self.dp.X_w2id['<GO>']
        self._x_eos = self.dp.X_w2id['<EOS>']
        self._x_pad = self.dp.X_w2id['<PAD>']
        self._x_unk = self.dp.X_w2id['<UNK>']

        self._y_go = self.dp.Y_w2id['<GO>']
        self._y_eos = self.dp.Y_w2id['<EOS>']
        self._y_pad = self.dp.Y_w2id['<PAD>']
        self._y_unk = self.dp.Y_w2id['<UNK>']

    def xToz(self, input_word):
        # print(input_word)
        input_word = input_word.split()
        # print(input_word)
        input_indices = [self.dp.X_w2id.get(char, self._x_unk) for char in input_word]
        # print(input_indices)
        z = self.sess.run(self.z,
                          {self.X: [input_indices], self.X_seq_len: [len(input_indices)], self.output_keep_prob: 1,
                           self.input_keep_prob: 1})
        return z

    def xTozmu(self, input_word):
        # print(input_word)
        input_word = input_word.split()
        # print(input_word)
        input_indices = [self.dp.X_w2id.get(char, self._x_unk) for char in input_word]
        # print(input_indices)
        z = self.sess.run(self.z_mu,
                          {self.X: [input_indices], self.X_seq_len: [len(input_indices)], self.output_keep_prob: 1,
                           self.input_keep_prob: 1})
        return z

    def xToz_grad(self, input_word, c):
        c_ = [0 for _ in range(self.num_classes)]
        c_[c] = 1
        # print(input_word)
        input_word = input_word.split()
        # print(input_word)
        input_indices = [self.dp.X_w2id.get(char, self._x_unk) for char in input_word]
        # print(input_indices)
        grad, z = self.sess.run([self.Dgrad, self.z],
                                {self.C: [c_], self.X: [input_indices], self.X_seq_len: [len(input_indices)],
                                 self.output_keep_prob: 1, self.input_keep_prob: 1})
        return z, grad

    def xTozmu_grad(self, input_word, c):
        c_ = [0 for _ in range(self.num_classes)]
        c_[c] = 1
        # print(input_word)
        input_word = input_word.split()
        # print(input_word)
        input_indices = [self.dp.X_w2id.get(char, self._x_unk) for char in input_word]
        # print(input_indices)
        grad, z = self.sess.run([self.Dgrad, self.z_mu],
                                {self.C: [c_], self.X: [input_indices], self.X_seq_len: [len(input_indices)],
                                 self.output_keep_prob: 1, self.input_keep_prob: 1})
        return z, grad

    def xTozmu_gradmu(self, input_word, c):
        c_ = [0 for _ in range(self.num_classes)]
        c_[c] = 1
        # print(input_word)
        input_word = input_word.split()
        # print(input_word)
        input_indices = [self.dp.X_w2id.get(char, self._x_unk) for char in input_word]
        # print(input_indices)
        grad, z = self.sess.run([self.Dgradmu, self.z_mu],
                                {self.C: [c_], self.X: [input_indices], self.X_seq_len: [len(input_indices)],
                                 self.output_keep_prob: 1, self.input_keep_prob: 1})
        return z, grad

    def zmu_gradCmu(self, z_mu, c):
        c_ = [0 for _ in range(self.num_classes)]
        c_[c] = 1
        grad = self.sess.run(self.Dgradmu,
                             {self.C: [c_], self.z: z_mu, self.output_keep_prob: 1, self.input_keep_prob: 1})
        return grad

    def zmu_gradBowCmu(self, z_mu, c, content):
        c_ = [0 for _ in range(self.num_classes)]
        c_[c] = 1
        content = content.split()
        content_indices = [self.dp.X_w2id.get(char, self._x_unk) for char in content]
        grad, bgrad = self.sess.run([self.Dgrad, self.Bowgradmu], {self.Y: [content_indices],
                                                                   self.C: [c_],
                                                                   self.z: z_mu,
                                                                   self.Y_seq_len: [len(content_indices)],
                                                                   self.output_keep_prob: 1, self.input_keep_prob: 1})
        return grad, bgrad

    def idxTozmu_batch(self, input_indices):
        input_indices_pad, length_list = self.pad(input_indices)
        z = self.sess.run(self.z_mu, {self.X: input_indices_pad, self.X_seq_len: length_list,
                                      self.output_keep_prob: 1, self.input_keep_prob: 1})
        return z

    def zmuTox_batch(self, z_mu_batch):
        out_indices = self.sess.run(self.predicting_ids, {self.batch_size: z_mu_batch.shape[0],
                                                          self.z: z_mu_batch, self.output_keep_prob: 1,
                                                          self.input_keep_prob: 1})
        outputs = []
        outputs_idx = []
        for idx in range(out_indices.shape[0]):
            eos_id = self.dp.Y_w2id['<EOS>']
            ot = out_indices[idx, :, 0]
            ot = ot.tolist()
            if eos_id in ot:
                ot = ot[:ot.index(eos_id)]
            if self.reverse:
                ot = ot[::-1]
            output_str = ' '.join([self.dp.Y_id2w.get(i, u'&') for i in ot])
            outputs_idx.append(ot)
            outputs.append(output_str)
        return outputs, outputs_idx

    def xTozmu_gradBowmu(self, input_word, content):
        input_word = input_word.split()
        content = content.split()
        # print(input_word)
        content_indices = [self.dp.X_w2id.get(char, self._x_unk) for char in content]
        input_indices = [self.dp.X_w2id.get(char, self._x_unk) for char in input_word]
        # print(input_indices)
        grad, z = self.sess.run([self.Bowgradmu, self.z_mu], {self.Y: [content_indices],
                                                              self.X: [input_indices],
                                                              self.Y_seq_len: [len(content_indices)],
                                                              self.X_seq_len: [len(input_indices)],
                                                              self.output_keep_prob: 1, self.input_keep_prob: 1})
        return z, grad

    def xTozmu_gradBowCmu(self, input_word, c, content):
        c_ = [0 for _ in range(self.num_classes)]
        c_[c] = 1
        input_word = input_word.split()
        content = content.split()
        # print(input_word)
        content_indices = [self.dp.X_w2id.get(char, self._x_unk) for char in content]
        input_indices = [self.dp.X_w2id.get(char, self._x_unk) for char in input_word]
        # print(input_indices)
        dgrad, grad, z = self.sess.run([self.Dgrad, self.Bowgradmu, self.z_mu], {self.Y: [content_indices],
                                                                                 self.X: [input_indices],
                                                                                 self.C: [c_],
                                                                                 self.Y_seq_len: [len(content_indices)],
                                                                                 self.X_seq_len: [len(input_indices)],
                                                                                 self.output_keep_prob: 1,
                                                                                 self.input_keep_prob: 1})
        return z, dgrad, grad

    def zTograd(self, z, c):
        c_ = [0 for _ in range(self.num_classes)]
        c_[c] = 1
        grad = self.sess.run(self.Dgrad, {self.C: [c_], self.z: z, self.output_keep_prob: 1, self.input_keep_prob: 1})
        return grad

    def zTox(self, z):
        out_indices, c, auc = self.sess.run([self.predicting_ids, self.predictions, self.ypred_for_auc],
                                            {self.batch_size: z.shape[0],
                                             self.z: z, self.output_keep_prob: 1, self.input_keep_prob: 1})
        outputs = []
        for idx in range(out_indices.shape[-1]):
            eos_id = self.dp.Y_w2id['<EOS>']
            ot = out_indices[0, :, idx]
            ot = ot.tolist()
            if eos_id in ot:
                ot = ot[:ot.index(eos_id)]
            if self.reverse:
                ot = ot[::-1]
            output_str = ' '.join([self.dp.Y_id2w.get(i, u'&') for i in ot])
            outputs.append(output_str)
        return outputs, c[0], auc[0]

    def generate(self, batch_size=6):
        out_indices, c_lists = self.sess.run([self.predicting_ids, self.predictions], {self.batch_size: batch_size,
                                                                                       self.z: np.random.randn(
                                                                                           batch_size, self.latent_dim),
                                                                                       self.output_keep_prob: 1,
                                                                                       self.input_keep_prob: 1})
        outputs = []
        for idx in range(out_indices.shape[0]):
            eos_id = self.dp.Y_w2id['<EOS>']
            ot = out_indices[idx, :, 0]  # The 0th beam of each batch
            ot = ot.tolist()
            if eos_id in ot:
                ot = ot[:ot.index(eos_id)]
            if self.reverse:
                ot = ot[::-1]
            output_str = ' '.join([self.dp.Y_id2w.get(i, u'&') for i in ot])
            outputs.append(output_str)
        return outputs, c_lists

    def infer(self, input_word):
        if self.reverse:
            input_word = input_word[::-1]
        input_indices = [self.dp.X_w2id.get(char, self._x_unk) for char in input_word.split()]
        out_indices = self.sess.run(self.predicting_ids, {
            self.X: [input_indices], self.X_seq_len: [len(input_indices)], self.output_keep_prob: 1,
            self.input_keep_prob: 1})
        outputs = []
        for idx in range(out_indices.shape[-1]):
            eos_id = self.dp.Y_w2id['<EOS>']
            ot = out_indices[0, :, idx]
            ot = ot.tolist()
            if eos_id in ot:
                ot = ot[:ot.index(eos_id)]
            if self.reverse:
                ot = ot[::-1]
            output_str = ' '.join([self.dp.Y_id2w.get(i, u'&') for i in ot])
            outputs.append(output_str)
        return outputs

    def infer_with_c(self, input_word):
        input_indices = [self.dp.X_w2id.get(char, self._x_unk) for char in input_word.split()]
        out_indices, C = self.sess.run([self.predicting_ids, self.predictions], {
            self.X: [input_indices], self.X_seq_len: [len(input_indices)], self.output_keep_prob: 1,
            self.input_keep_prob: 1})
        outputs = []
        for idx in range(out_indices.shape[-1]):
            eos_id = self.dp.Y_w2id['<EOS>']
            ot = out_indices[0, :, idx]
            ot = ot.tolist()
            if eos_id in ot:
                ot = ot[:ot.index(eos_id)]
            if self.reverse:
                ot = ot[::-1]
            output_str = ' '.join([self.dp.Y_id2w.get(i, u'&') for i in ot])
            outputs.append(output_str)
        return outputs, C[0]

    def infer_with_c_batch(self, input_word_batch):
        input_indices_batch = np.array(
            [[self.dp.X_w2id.get(char, self._x_unk) for char in input_word.split()] for input_word in input_word_batch])
        input_indices_batch, input_indices_lengths = self.dp.pad_sentence_batch(input_indices_batch, self.dp._y_pad)
        out_indices, C = self.sess.run([self.predicting_ids, self.predictions], {
            self.X: input_indices_batch, self.X_seq_len: input_indices_lengths, self.output_keep_prob: 1,
            self.input_keep_prob: 1})
        outputs = []
        for idx in range(out_indices.shape[0]):
            eos_id = self.dp.Y_w2id['<EOS>']
            ot = out_indices[idx, :, 0]
            ot = ot.tolist()
            if eos_id in ot:
                ot = ot[:ot.index(eos_id)]
            if self.reverse:
                ot = ot[::-1]
            output_str = ' '.join([self.dp.Y_id2w.get(i, u'&') for i in ot])
            outputs.append(output_str)
        return outputs, C

    def prefix_infer(self, input_word, prefix):
        input_indices_X = [self.dp.X_w2id.get(char, self._x_unk) for char in input_word.split()]
        input_indices_Y = [self.dp.Y_w2id.get(char, self._y_unk) for char in prefix.split()]
        prefix_go = []
        prefix_go.append(input_indices_Y[-1])
        out_indices, scores = self.sess.run([self.predicting_prefix_ids, self.prefix_score], {
            self.X: [input_indices_X], self.X_seq_len: [len(input_indices_X)], self.Y: [input_indices_Y],
            self.Y_seq_len: [len(input_indices_Y)],
            self.prefix_go: prefix_go, self.input_keep_prob: 1, self.output_keep_prob: 1})

        outputs = []
        for idx in range(out_indices.shape[-1]):
            eos_id = self.dp.Y_w2id['<EOS>']
            ot = out_indices[0, :, idx]
            ot = ot.tolist()
            if eos_id in ot:
                ot = ot[:ot.index(eos_id)]
                if self.reverse:
                    ot = ot[::-1]
            if self.reverse:
                output_str = ' '.join([self.dp.Y_id2w.get(i, u'&') for i in ot]) + ' ' + prefix
            else:
                output_str = prefix + ' ' + ' '.join([self.dp.Y_id2w.get(i, u'&') for i in ot])
            outputs.append(output_str)
        return outputs

    def pad(self, x, move_go=False):
        if move_go:
            length_list = [len(k) - 1 for k in x]
        else:
            length_list = [len(k) for k in x]
        max_length = max(length_list)
        pad_x = []
        for k in x:
            if move_go:
                pad_k = k[1:] + [self.dp.X_w2id['<PAD>'], ] * (max_length - len(k[1:]))
            else:
                pad_k = k + [self.dp.X_w2id['<PAD>'], ] * (max_length - len(k))
            pad_x.append(pad_k)
        return pad_x, length_list

    def restore(self, path):
        self.saver.restore(self.sess, path)
        print('restore %s success' % path)

    def get_learning_rate_decay(self, decay_scheme='luong234'):
        num_train_steps = self.dp.num_steps
        if decay_scheme == "luong10":
            start_decay_step = int(num_train_steps / 2)
            remain_steps = num_train_steps - start_decay_step
            decay_steps = int(remain_steps / 10)  # decay 10 times
            decay_factor = 0.5
        else:
            start_decay_step = int(num_train_steps * 2 / 3)
            remain_steps = num_train_steps - start_decay_step
            decay_steps = int(remain_steps / 4)  # decay 4 times
            decay_factor = 0.5
        return tf.cond(
            self.global_step < start_decay_step,
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                (self.global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")

    def setup_summary(self):
        train_loss = tf.Variable(0.)
        tf.summary.scalar('Train_loss', train_loss)

        test_loss = tf.Variable(0.)
        tf.summary.scalar('Test_loss', test_loss)

        bleu_score = tf.Variable(0.)
        tf.summary.scalar('BLEU_score', bleu_score)

        tf.summary.scalar('lr_rate', self.learning_rate)

        summary_vars = [train_loss, test_loss, bleu_score]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


class VAE_DP:
    def __init__(self, X_indices, Y_indices, C_labels, X_w2id, Y_w2id, BATCH_SIZE, n_epoch, split_ratio=0.1,
                 is_shuffle=False, test_data=None):
        self.n_epoch = n_epoch
        if test_data == None:
            num_test = int(len(X_indices) * split_ratio)
            r = np.random.permutation(len(X_indices))
            X_indices = np.array(X_indices)[r].tolist()
            Y_indices = np.array(Y_indices)[r].tolist()
            C_labels = np.array(C_labels)[r].tolist()
            self.C_train = np.array(C_labels[num_test:])
            self.X_train = np.array(X_indices[num_test:])
            self.Y_train = np.array(Y_indices[num_test:])
            self.C_test = np.array(C_labels[:num_test])
            self.X_test = np.array(X_indices[:num_test])
            self.Y_test = np.array(Y_indices[:num_test])
        else:
            self.X_train, self.Y_train, self.C_train, self.X_test, self.Y_test, self.C_test = test_data
            self.X_train = np.array(self.X_train)
            self.Y_train = np.array(self.Y_train)
            self.C_train = np.array(self.C_train)
            self.X_test = np.array(self.X_test)
            self.Y_test = np.array(self.Y_test)
            self.C_test = np.array(self.C_test)

        assert len(self.X_train) == len(self.Y_train)
        self.num_batch = int(len(self.X_train) / BATCH_SIZE)
        self.is_shuffle = is_shuffle
        self.num_steps = self.num_batch * self.n_epoch
        self.batch_size = BATCH_SIZE
        self.X_w2id = X_w2id
        self.X_id2w = dict(zip(X_w2id.values(), X_w2id.keys()))
        self.Y_w2id = Y_w2id
        self.Y_id2w = dict(zip(Y_w2id.values(), Y_w2id.keys()))
        self._x_pad = self.X_w2id['<PAD>']
        self._y_pad = self.Y_w2id['<PAD>']
        print(
            'Train_data: %d | Test_data: %d | Batch_size: %d | Num_batch: %d | X_vocab_size: %d | Y_vocab_size: %d' % (
            len(self.X_train), len(self.X_test), BATCH_SIZE, self.num_batch, len(self.X_w2id), len(self.Y_w2id)))

    def next_batch(self, X, Y, C):
        r = np.random.permutation(len(X))
        X = X[r]
        Y = Y[r]
        C = C[r]

        for i in range(0, len(X) - len(X) % self.batch_size, self.batch_size):
            if self.is_shuffle:
                X_batch = []
                for x in X[i: i + self.batch_size]:
                    a = [t for t in range(len(x))]
                    for j in range(len(a)):
                        a[j] += np.random.randint(0, 4)
                    p = np.argsort(a)
                    # print(' '.join([id2w[idx] for idx in x]))
                    x = np.array(copy.deepcopy(x))[p].tolist()
                    # print(' '.join([id2w[idx] for idx in x]))
                    X_batch.append(x)
                X_batch = np.array(X_batch)
            else:
                X_batch = X[i: i + self.batch_size]

            Y_batch = Y[i: i + self.batch_size]
            C_batch = C[i: i + self.batch_size]

            padded_X_batch, X_batch_lens = self.pad_sentence_batch(X_batch, self._x_pad)
            padded_Y_batch, Y_batch_lens = self.pad_sentence_batch(Y_batch, self._y_pad)
            yield (np.array(padded_X_batch),
                   np.array(padded_Y_batch),
                   C_batch,
                   X_batch_lens,
                   Y_batch_lens)

    def sample_test_batch(self):
        C = self.C_test[:self.batch_size]
        padded_X_batch, X_batch_lens = self.pad_sentence_batch(self.X_test[: self.batch_size], self._x_pad)
        padded_Y_batch, Y_batch_lens = self.pad_sentence_batch(self.Y_test[: self.batch_size], self._y_pad)
        return np.array(padded_X_batch), np.array(padded_Y_batch), C, X_batch_lens, Y_batch_lens

    def pad_sentence_batch(self, sentence_batch, pad_int):
        padded_seqs = []
        seq_lens = []
        sentence_batch = sentence_batch.tolist()
        max_sentence_len = max([len(sentence) for sentence in sentence_batch])
        for sentence in sentence_batch:
            padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
            seq_lens.append(len(sentence))
        return padded_seqs, seq_lens


import scipy.interpolate as si
from scipy import interpolate


def pad(x, pid, move_go=False):
    x = [k[:30] for k in x]
    if move_go:
        length_list = [len(k) - 1 for k in x]
    else:
        length_list = [len(k) for k in x]
    max_length = max(length_list)
    pad_x = []
    for k in x:
        if move_go:
            pad_k = k[1:] + [pid, ] * (max_length - len(k[1:]))
        else:
            pad_k = k + [pid, ] * (max_length - len(k))
        pad_x.append(pad_k)
    return pad_x, length_list


def pad_maxlength(x, pid, move_go=False):
    max_length = 30
    if move_go:
        length_list = [len(k) - 1 for k in x]
    else:
        length_list = [min(len(k), max_length) for k in x]

    pad_x = []
    for k in x:
        if move_go:
            pad_k = k[1:] + [pid, ] * (max_length - len(k[1:]))
        else:
            pad_k = k[:max_length] + [pid, ] * (max_length - len(k))
        pad_x.append(pad_k)
    return pad_x, length_list


def BetaGenerator(epoches, beta_decay_period, beta_decay_offset):
    points = [[0, 0], [0, beta_decay_offset], [0, beta_decay_offset + 0.33 * beta_decay_period],
              [1, beta_decay_offset + 0.66 * beta_decay_period], [1, beta_decay_offset + beta_decay_period],
              [1, epoches]];
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    t = range(len(points))
    ipl_t = np.linspace(0.0, len(points) - 1, 100)
    x_tup = si.splrep(t, x, k=3)
    y_tup = si.splrep(t, y, k=3)
    x_list = list(x_tup)
    xl = x.tolist()
    x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]
    y_list = list(y_tup)
    yl = y.tolist()
    y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]
    x_i = si.splev(ipl_t, x_list)
    y_i = si.splev(ipl_t, y_list)
    return interpolate.interp1d(y_i, x_i)


class VAE_util:
    def __init__(self, dp, model, display_freq=3, is_show=True):
        self.display_freq = display_freq
        self.is_show = is_show
        self.dp = dp
        self.model = model
        self.betaG = BetaGenerator(self.dp.n_epoch * self.dp.num_batch,
                                   self.model.beta_decay_period * self.dp.num_batch,
                                   self.model.beta_decay_offset * self.dp.num_batch)

    def train(self, epoch):
        avg_r_loss = 0.0
        avg_c_loss = 0.0
        avg_kl_loss = 0.0
        avg_acc = 0.0
        tic = time.time()
        X_test_batch, Y_test_batch, C_test_batch, X_test_batch_lens, Y_test_batch_lens = self.dp.sample_test_batch()
        for local_step, (
        X_train_batch, Y_train_batch, C_train_batch, X_train_batch_lens, Y_train_batch_lens) in enumerate(
                self.dp.next_batch(self.dp.X_train, self.dp.Y_train, self.dp.C_train)):
            # print(len(C_train_batch), len(X_train_batch))
            beta = 0.001 + self.betaG(self.model.step)
            self.model.step, _, r_loss, c_loss, acc, kl_loss = self.model.sess.run(
                [self.model.global_step, self.model.train_op, self.model.r_loss, self.model.c_loss, self.model.accuracy,
                 self.model.kl_loss],
                {self.model.X: X_train_batch,
                 self.model.Y: Y_train_batch,
                 self.model.C: C_train_batch,
                 self.model.X_seq_len: X_train_batch_lens,
                 self.model.Y_seq_len: Y_train_batch_lens,
                 self.model.output_keep_prob: self.model._output_keep_prob,
                 self.model.input_keep_prob: self.model._input_keep_prob,
                 self.model.B: beta})
            avg_r_loss += r_loss
            avg_c_loss += c_loss
            avg_kl_loss += kl_loss
            avg_acc += acc
            """
            stats = [loss]
            for i in xrange(len(stats)):
                self.model.sess.run(self.model.update_ops[i], feed_dict={
                    self.model.summary_placeholders[i]: float(stats[i])
                })
            summary_str = self.model.sess.run([self.model.summary_op])
            self.summary_writer.add_summary(summary_str, self.model.step + 1)
            """
            if self.is_show:
                if (local_step % int(self.dp.num_batch / self.display_freq)) == 0:
                    val_r_loss, val_c_loss, val_acc, val_kl_loss = self.model.sess.run(
                        [self.model.r_loss, self.model.c_loss, self.model.accuracy, self.model.kl_loss],
                        {self.model.X: X_test_batch,
                         self.model.Y: Y_test_batch,
                         self.model.C: C_test_batch,
                         self.model.X_seq_len: X_test_batch_lens,
                         self.model.Y_seq_len: Y_test_batch_lens,
                         self.model.output_keep_prob: 1,
                         self.model.input_keep_prob: 1,
                         self.model.B: beta})
                    print(
                        "Epoch %d/%d | Batch %d/%d | Train_loss: R %.3f C %.3f acc %.3f kl %.3f | Test_loss: R %.3f C %.3f acc %.3f kl %.3f | Time_cost:%.3f" % (
                        epoch, self.n_epoch, local_step, self.dp.num_batch,
                        avg_r_loss / (local_step + 1),
                        avg_c_loss / (local_step + 1),
                        avg_acc / (local_step + 1),
                        avg_kl_loss / (local_step + 1),
                        val_r_loss,
                        val_c_loss,
                        val_acc,
                        val_kl_loss,
                        time.time() - tic))
                    self.cal()

                    tic = time.time()
        return avg_r_loss / (local_step + 1), avg_c_loss / (local_step + 1), avg_acc / (local_step + 1), avg_kl_loss / (
                    local_step + 1)

    def test(self):
        avg_r_loss = 0.0
        avg_c_loss = 0.0
        avg_kl_loss = 0.0
        avg_acc = 0.0
        beta = 0.001 + self.betaG(self.model.step)
        for local_step, (X_test_batch, Y_test_batch, C_test_batch, X_test_batch_lens, Y_test_batch_lens) in enumerate(
                self.dp.next_batch(self.dp.X_test, self.dp.Y_test, self.dp.C_test)):
            r_loss, c_loss, acc, kl_loss = self.model.sess.run(
                [self.model.r_loss, self.model.c_loss, self.model.accuracy, self.model.kl_loss],
                {self.model.X: X_test_batch,
                 self.model.Y: Y_test_batch,
                 self.model.C: C_test_batch,
                 self.model.X_seq_len: X_test_batch_lens,
                 self.model.Y_seq_len: Y_test_batch_lens,
                 self.model.output_keep_prob: 1,
                 self.model.input_keep_prob: 1,
                 self.model.B: beta})
            avg_r_loss += r_loss
            avg_c_loss += c_loss
            avg_kl_loss += kl_loss
            avg_acc += acc
        return avg_r_loss / (local_step + 1), avg_c_loss / (local_step + 1), avg_acc / (local_step + 1), avg_kl_loss / (
                    local_step + 1)

    def fit(self, train_dir, is_bleu):
        self.n_epoch = self.dp.n_epoch
        out_dir = train_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print("Writing to %s" % out_dir)
        checkpoint_prefix = os.path.join(out_dir, "model")
        self.summary_writer = tf.summary.FileWriter(os.path.join(out_dir, 'Summary'), self.model.sess.graph)
        for epoch in range(1, self.n_epoch + 1):
            tic = time.time()
            train_r_loss, train_c_loss, train_acc, train_kl = self.train(epoch)
            test_r_loss, test_c_loss, test_acc, test_kl = self.test()

            print(
                "Epoch %d/%d | Train_loss: R %.3f C %.3f acc %.3f kl %.3f | Test_loss: R %.3f C %.3f acc %.3f kl %.3f " % (
                epoch, self.n_epoch, train_r_loss, train_c_loss, train_acc, train_kl,
                test_r_loss, test_c_loss, test_acc, test_kl))
            path = self.model.saver.save(self.model.sess, checkpoint_prefix, global_step=epoch)
            print("Saved model checkpoint to %s" % path)

    def show(self, sent, id2w):
        return " ".join([id2w.get(idx, u'&') for idx in sent])

    def cal(self, n_example=5):
        train_n_example = int(n_example / 2)
        test_n_example = n_example - train_n_example
        for _ in range(test_n_example):
            example = self.show(self.dp.X_test[_], self.dp.X_id2w)
            y = self.show(self.dp.Y_test[_], self.dp.Y_id2w)
            o, c = self.model.infer_with_c(example)
            o = o[0]
            print('Input: %s | Output: %s %d | GroundTruth: %s %d' % (example, o, c, y, np.argmax(self.dp.C_test[_])))
        for _ in range(train_n_example):
            example = self.show(self.dp.X_train[_], self.dp.X_id2w)
            y = self.show(self.dp.Y_train[_], self.dp.Y_id2w)
            o = self.model.infer(example)[0]
            print('Input: %s | Output: %s %d | GroundTruth: %s %d' % (example, o, c, y, np.argmax(self.dp.C_train[_])))
        print("")

    def test_bleu(self, N=300, gram=4):
        all_score = []
        for i in range(N):
            input_indices = self.show(self.dp.X_test[i], self.dp.X_id2w)
            o = self.model.infer(input_indices)[0]
            refer4bleu = [[' '.join([self.dp.Y_id2w.get(w, u'&') for w in self.dp.Y_test[i]])]]
            candi = [' '.join(w for w in o)]
            score = mybleu.BLEU(candi, refer4bleu, gram=gram)
            all_score.append(score)
        return np.mean(all_score)

    def show_res(self, path):
        res = cPickle.load(open(path))
        plt.figure(1)
        plt.title('The results')
        l1, = plt.plot(res[0], 'g')
        l2, = plt.plot(res[1], 'r')
        l3, = plt.plot(res[3], 'b')
        plt.legend(handles=[l1, l2, l3], labels=["Train_loss", "Test_loss", "BLEU"], loc='best')
        plt.show()

    def test_all(self, path, epoch_range, is_bleu=True):
        val_loss_list = []
        bleu_list = []
        for i in range(epoch_range[0], epoch_range[-1]):
            self.model.restore(path + str(i))
            val_loss = self.test()
            val_loss_list.append(val_loss)
            if is_bleu:
                bleu_score = self.test_bleu()
                bleu_list.append(bleu_score)
        plt.figure(1)
        plt.title('The results')
        l1, = plt.plot(val_loss_list, 'r')
        l2, = plt.plot(bleu_list, 'b')
        plt.legend(handles=[l1, l2], labels=["Test_loss", "BLEU"], loc='best')
        plt.show()

########################
## Data
#######################
w2id, id2w = pickle.load(open('shakespearian-corpus/w2id_id2w.pkl','rb'))
Y_train, C_train = pickle.load(open('shakespearian-corpus/train_C.pkl','rb'))
Y_dev, C_dev = pickle.load(open('shakespearian-corpus/valid_C.pkl','rb'))
Y_test, C_test = pickle.load(open('shakespearian-corpus/test_C.pkl','rb'))

X_train = [x[:-1] for x in Y_train]
X_dev = [x[:-1] for x in Y_dev]
X_test = [x[:-1] for x in Y_test]

################################
##  Set Parameters for training
################################
BATCH_SIZE = 200
NUM_EPOCH = 35
is_shuffle = False
Latent_weight = 0.4
Model_basic_name = 'VAE-All'
train_dir ='model/shakespeare-to-modern/VAE/' + Model_basic_name
vae_dp = VAE_DP(None, None, None, w2id, w2id, BATCH_SIZE, test_data=(X_train, Y_train, C_train, X_dev, Y_dev, C_dev), n_epoch=NUM_EPOCH, is_shuffle=is_shuffle)

is_training = False

############################
##  Training
###########################
is_training = True
if is_training:
    g = tf.Graph()
    sess = tf.Session(graph=g, config=sess_conf)
    with sess.as_default():
        with sess.graph.as_default():
            model = VAE(
                dp = vae_dp,
                rnn_size = 512,
                n_layers = 1,
                encoder_embedding_dim = 128,
                decoder_embedding_dim = 128,
                cell_type = 'lstm',
                latent_dim = 512,
                beta_decay_period = 10,
                beta_decay_offset = 5,
                latent_weight = Latent_weight,
                bow_size = 400,
                is_inference = False,
                num_classes = 2,
                max_infer_length = 20,
                #att_type='B',
                beam_width=10,
                Lambda = 0.9,
                gamma = 10.0,
                residual = False,
                sess=sess
            )
            #print(len(tf.global_variables()))

    util = VAE_util(dp=vae_dp, model=model)
    #model.restore('model/shakespeare-to-modern/VAE/model-35')
    util.fit(train_dir=train_dir, is_bleu=False)
