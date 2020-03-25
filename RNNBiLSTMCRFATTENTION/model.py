import time
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib import rnn
from general_utils import get_logger , Progbar
from utils import get_chunks
from datapreprocess import pad_sequences, batch_yield, get_train_test_data
from tensorflow.contrib.seq2seq import BahdanauAttention, TrainingHelper
from tensorflow.contrib.seq2seq import AttentionWrapper, BasicDecoder, dynamic_decode


class MF_SequenceLabelingModel(object):


    def __init__(self,
                 feature_embedding_list,
                 feature_num,
                 feature_weight_dropout_list,
                 label2id,num_class,
                 batch_size,
                 epoch_num,max_patience,
                 num_layers,
                 rnn_unit,hidden_dim,
                 dropout,
                 optimizer,
                 lr,clip,
                 use_crf,
                 output_path,
                 is_attention,
                 config,):
        #数据

        self.feature_embedding_list=feature_embedding_list
        self.feature_num=feature_num
        self.weight_dropout_list=feature_weight_dropout_list
        self.label2id = label2id
        self.num_class = num_class
        #模型参数
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.max_patience = max_patience

        self.num_layers = num_layers #bilstm layers
        self.rnn_unit = rnn_unit
        self.hidden_dim = hidden_dim

        self.dropout = dropout

        self.optimizer= optimizer
        self.lr = lr
        self.clip = clip

        self.use_crf= use_crf
        self.is_attention = is_attention
        self.outputpath= output_path
        self.config = config

        self.model_path = os.path.join(self.outputpath, "checkpoints/")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        result_path = os.path.join(self.outputpath, "results")
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        log_path = os.path.join(result_path, "log.txt")
        self.logger = get_logger(log_path)
        # self.logger.info(self.config)


        self.build()
        # self.logger.info("best score acc:", str(91))
        # self.logger.info(json.dumps(self.config, indent=1))





    def add_placeholders(self):
        """
        Adds placeholders to self
        """
        self.input_feature_ph_list=[]
        self.weight_dropout_ph_list=[]
        for i in range(self.feature_num):
            self.input_feature_ph_list.append(tf.placeholder(tf.int32, shape=[None, None], name='input_feature_ph_%d' % i))
            self.weight_dropout_ph_list.append(tf.placeholder(dtype=tf.float32, shape=[], name="weight_dropout_%d" %i))
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def add_word_embeddings_op(self):
        """
        _embedding的维度为词向量的维度[word_num, embedding_len]
        embeddings的维度为输入特征的维度[None, None]转为词向量的维度[None, None, embedding_len]
        feature_embeddings为n个输入特征+dropout的列表
        input_feature_embeddings为n个输入特征的按照axis=2的拼接形成的矩阵
        Adds word embeddings to self
        """
        self.feature_embeddings = []
        for i in range(self.feature_num):
            # trainable= not self.config['model_params']['embed_params'][i]['pre_train']
            _embeddings = tf.Variable(self.feature_embedding_list[i],
                                           dtype=tf.float32,
                                           trainable=True,
                                           )
            embeddings = tf.nn.embedding_lookup(params=_embeddings,
                                                     ids=self.input_feature_ph_list[i],
                                                     )
            self.feature_embeddings.append(tf.nn.dropout(embeddings, self.weight_dropout_ph_list[i]))
        # concat all features
        self.input_feature_embeddings = self.feature_embeddings[0] if self.feature_num == 1 \
                else tf.concat(values=self.feature_embeddings, axis=2, name='input_features')


    def add_multilayer_rnn_op(self):
        """
        Adds logits to self
        """
        with tf.variable_scope("bi-lstm"):
            _inputs = self.input_feature_embeddings
            for n in range(self.num_layers):
                with tf.variable_scope(None, default_name="bidirectional-rnn"):
                    if self.rnn_unit == 'lstm':
                        cell_fw = rnn.LSTMCell(self.hidden_dim, forget_bias=1., state_is_tuple=True)
                        cell_bw = rnn.LSTMCell(self.hidden_dim, forget_bias=1., state_is_tuple=True)
                    elif self.rnn_unit == 'gru':
                        cell_fw = rnn.GRUCell(self.hidden_dim)
                        cell_bw = rnn.GRUCell(self.hidden_dim)
                    elif self.rnn_unit == 'rnn':
                        cell_fw = rnn.BasicRNNCell(self.hidden_dim)
                        cell_bw = rnn.BasicRNNCell(self.hidden_dim)
                    else:
                        raise ValueError('rnn_unit must in (lstm, gru, rnn)!')

                    initial_state_fw = cell_fw.zero_state(tf.shape(self.input_feature_embeddings)[0], dtype=tf.float32)
                    initial_state_bw = cell_bw.zero_state(tf.shape(self.input_feature_embeddings)[0], dtype=tf.float32)
                    (output, state) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, _inputs, self.sequence_lengths,
                                                                      initial_state_fw, initial_state_bw, dtype=tf.float32)
                    _inputs = tf.concat(output, 2)
            self.output = tf.nn.dropout(_inputs, self.dropout_pl)

        if self.is_attention:
            with tf.variable_scope('attention'):
                embedding_dim = self.hidden_dim  * 2
                attn_mech = BahdanauAttention(embedding_dim, _inputs, self.sequence_lengths)
                dec_cell = rnn.LSTMCell(self.hidden_dim, state_is_tuple=True)
                attn_cell = AttentionWrapper(dec_cell, attn_mech, embedding_dim)
                attn_zero = attn_cell.zero_state(tf.shape(self.input_feature_embeddings)[0], dtype=tf.float32)
                helper = TrainingHelper(
                    inputs=_inputs,
                    sequence_length=self.sequence_lengths)
                decoder = BasicDecoder(cell=attn_cell, helper=helper, initial_state=attn_zero)
                final_outputs, final_state, final_sequence_length = dynamic_decode(decoder)

            self.output = tf.nn.dropout(final_outputs.rnn_output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", shape=[2 * self.hidden_dim, self.num_class],
                                dtype=tf.float32)

            b = tf.get_variable("b", shape=[self.num_class], dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            s = tf.shape(self.output)
            output = tf.reshape(self.output, [-1, 2 * self.hidden_dim])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_class])


    def add_pred_op(self):
        """
        Adds labels_pred to self
        """
        if not self.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)


    def add_loss_op(self):
        """
        Adds loss to self
        """
        if self.use_crf:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                        tag_indices=self.labels,
                                                                        sequence_lengths=self.sequence_lengths)
            
            self.loss = -tf.reduce_mean(log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def add_train_op(self):
        """
        Add train_op to self
        """
        with tf.variable_scope("train_step"):
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            gradients, variables = zip(*optim.compute_gradients(self.loss))
            gradients, global_norm = tf.clip_by_global_norm(gradients, self.clip)
            self.train_op = optim.apply_gradients(zip(gradients, variables))


    def add_init_op(self):
        init = tf.global_variables_initializer()
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(init)


    def add_summary(self):
        # tensorboard stuff
        if not os.path.exists(self.outputpath): os.makedirs(self.outputpath)
        summary_path = os.path.join(self.outputpath, "summaries")
        if not os.path.exists(summary_path): os.makedirs(summary_path)

        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(summary_path, self.sess.graph)


    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_multilayer_rnn_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()


    def predict_batch(self, sentence):
        fd, sequence_lengths = self.get_feed_dict(seqs_batch=sentence, weight_dropout=self.weight_dropout_list,dropout =self.dropout)


        if self.use_crf:
            viterbi_sequences = []
            logits, transition_params = self.sess.run([self.logits, self.transition_params],
                                                 feed_dict=fd)
            # iterate over the sentences
            for logit, sequence_length in zip(logits, sequence_lengths):
                # keep only the valid time steps
                logit = logit[:sequence_length]
                viterbi_sequence, viterbi_score = viterbi_decode(
                    logit, transition_params)
                viterbi_sequences += [viterbi_sequence]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths


    def run_epoch(self, train, dev, epoch):
        nbatches = (len(train) + self.batch_size - 1) // self.batch_size

        prog = Progbar(target=nbatches) #进度条

        batches = batch_yield(train, self.batch_size)
        for i, (sentence, label) in enumerate(batches):
            fd, _ = self.get_feed_dict(sentence, self.weight_dropout_list,label, self.lr,self.dropout )

            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])


            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)

        acc, p,r,f1 = self.run_evaluate(dev)
        self.logger.info("- dev acc {:04.2f} - p {:04.2f}- r {:04.2f}- f1 {:04.2f}".format(100 * acc,100 * p,100 * r, 100 * f1))
        return acc, p,r,f1


    def run_evaluate(self, test):
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.

        for words, labels in batch_yield(test, self.batch_size):

            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]
                lab_chunks = set(get_chunks(lab, self.label2id))
                lab_pred_chunks = set(get_chunks(lab_pred, self.label2id))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        return acc, p,r,f1


    def train(self,train_data, dev_data):
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('start_time:',start_time)
        self.add_summary()

        best_score = 0
        saver = tf.train.Saver()
        # for early stopping
        nepoch_no_imprv = 0

        # dev, train = get_train_test_data(data, 10)
        train = train_data
        dev = dev_data



        for epoch in range(self.epoch_num):

            self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.epoch_num))
            acc, p, r, f1 = self.run_epoch(train, dev, epoch)


            # early stopping and saving best parameters
            if acc >= best_score:
                nepoch_no_imprv = 0
                saver.save(self.sess, self.model_path)
                best_score = acc
                self.logger.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.max_patience:
                    self.logger.info("- early stopping {} epochs without improvement".format(
                        nepoch_no_imprv))
                    saver.save(self.sess, self.model_path)
                    break
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('end_time:',end_time)
        self.logger.info("best score acc: " + str(best_score))
        self.logger.info(json.dumps(self.config, indent=1))


    def test(self, test):
        saver = tf.train.Saver()

        self.logger.info("Testing model over test set")
        saver.restore(self.sess, self.model_path)
        acc,p,r, f1 = self.run_evaluate(test)
        self.logger.info("- test acc {:04.2f} -precision {:04.2f} - recall {:04.2f} - f1 {:04.2f}".format(100 * acc,100 * p,100 * r, 100 * f1))
        return acc,p,r, f1


    def get_feed_dict(self, seqs_batch,weight_dropout, labels=None, lr=None, dropout=None):

        feed_dict={}
        seq_len_list=0
        input_features=[]
        [input_features.append([]) for _ in range(self.feature_num)]


        for sen_i in range(len(seqs_batch)):
            for i in range(self.feature_num):
                input_features[i].append(seqs_batch[sen_i][i])
        for i in range(self.feature_num):
            input_feature,seq_len_list = pad_sequences(input_features[i], pad_mark=0)

            feed_dict[self.input_feature_ph_list[i]]=input_feature
            feed_dict[self.weight_dropout_ph_list[i]]=weight_dropout[i]

        feed_dict[self.sequence_lengths] = seq_len_list
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

