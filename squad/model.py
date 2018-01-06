# -*- coding: UTF-8 -*-
"""Define QA models here.
"""
import logging
import os
from copy import copy
from random import shuffle
from math import ceil

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

__author__ = 'Kensuke Muraki'

_LOGGER = logging.getLogger(__name__)


class HParams(object):

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class BaseQAModel(object):

    def __init__(self, params):
        self.params = params
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_graph()
            _LOGGER.info('The number of parameters = {:,}'.format(
                int(np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))))

    def fit(self, train_data, valid_data, train_dir, batch_size=10, epochs=10):
        sample_num = len(train_data['contexts'])
        best_f1 = 0.

        with tf.Session(graph=self.graph) as session:
            session.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(
                os.path.join(train_dir, 'tensorboard.log'), session.graph)

            for epoch_id in range(epochs):
                _LOGGER.info('--- Start epoch_id={} ---'.format(epoch_id))
                train_loss, train_start_accuracy, train_end_accuracy = self._train_epoch(
                    train_data, batch_size, sample_num, session)

                _LOGGER.info('Finished training epoch_id={}.\n'.format(epoch_id))
                _LOGGER.info('Train stats: loss={}, start_accuracy={}, end_accuracy={}\n'.format(
                    train_loss, train_start_accuracy, train_end_accuracy))

                valid_exact_match, valid_f1 = self._validate(valid_data, batch_size, session)

                _LOGGER.info('Finished validation.')
                _LOGGER.info('Validation stats: exact_match={}, f1={}\n'.format(
                    valid_exact_match, valid_f1))

                self._add_summary(epoch_id, session, summary_writer, train_end_accuracy, train_loss,
                                  train_start_accuracy, valid_exact_match, valid_f1)

                # save the model if get the best f1 score so far
                if valid_f1 > best_f1:
                    best_f1 = valid_f1
                    _LOGGER.info('Achieved the best f1 score so far. Saving the model...')
                    self.saver.save(session, os.path.join(train_dir, 'model.ckpt'))

    def _add_summary(self, epoch_id, session, summary_writer, train_end_accuracy, train_loss,
                     train_start_accuracy, valid_exact_match, valid_f1):
        metric_summary = tf.Summary()

        # add train summary
        metric_summary.value.add(tag='train_loss', simple_value=train_loss)
        metric_summary.value.add(tag='train_start_accuracy', simple_value=train_start_accuracy)
        metric_summary.value.add(tag='train_end_accuracy', simple_value=train_end_accuracy)

        # add validation summary
        metric_summary.value.add(tag='valid_exact_match', simple_value=valid_exact_match)
        metric_summary.value.add(tag='valid_f1', simple_value=valid_f1)

        summary_writer.add_summary(metric_summary, global_step=epoch_id)

        # add histogram summary of variables
        summary_writer.add_summary(session.run(self.variable_summary), epoch_id)

    def _validate(self, valid_data, batch_size, session):
        sample_num = len(valid_data['contexts'])
        batch_num = int(ceil(float(sample_num) / batch_size))
        exact_matches = []
        f1s = []

        for batch_id in range(batch_num):
            sample_start_id = batch_id * batch_size
            sample_end_id = min((sample_start_id + batch_size, sample_num))
            batch = self._get_batch(range(sample_start_id, sample_end_id), valid_data)
            batch = self._add_paddings(batch)
            answer_tuples = valid_data['spans'][sample_start_id: sample_end_id]

            start_probabilities, end_probabilities = session.run(
                [self.start_probabilities, self.end_probabilities],
                feed_dict={self.contexts: batch[0], self.questions: batch[1],
                           self.context_lens: batch[2], self.question_lens: batch[3]})

            predictions = self.search(start_probabilities, end_probabilities)
            batch_exact_matches, batch_f1s = self.compute_metrics(predictions, answer_tuples)

            exact_matches.extend(batch_exact_matches)
            f1s.extend(batch_f1s)

        return np.mean(exact_matches), np.mean(f1s)

    def compute_metrics(self, predictions, answer_tuples):

        def compute_f1(answer, prediction):
            true_positive = float(len(prediction.intersection(answer)))
            false_positive = float(len(prediction.difference(answer)))
            false_negative = float(len(answer.difference(prediction)))

            if (true_positive + false_positive == 0.) or (true_positive + false_negative == 0.):
                return 0.

            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)

            if precision == 0. and recall == 0:
                return 0.
            else:
                return 2 * precision * recall / (precision + recall)

        exact_matches = []
        f1s = []

        for prediction, answers in zip(predictions, answer_tuples):
            prediction_set = set(range(prediction[0], prediction[1] + 1))
            best_exact_match = 0
            best_f1 = 0.

            for answer in answers:
                exact_match = int(prediction == tuple(answer))
                best_exact_match = max(exact_match, best_exact_match)
                answer_set = set(range(answer[0], answer[1] + 1))
                f1 = compute_f1(answer_set, prediction_set)
                best_f1 = max(f1, best_f1)

            exact_matches.append(best_exact_match)
            f1s.append(best_f1)

        return exact_matches, f1s

    def _train_epoch(self, train_data, batch_size, sample_num, session):
        losses = []
        start_accuracies = []
        end_accuracies = []
        batch_num = int(ceil(float(sample_num) / batch_size))
        sample_ids = range(sample_num)
        shuffle(sample_ids)

        for batch_id in range(batch_num):
            sample_start_id = batch_id * batch_size
            sample_end_id = min((sample_start_id + batch_size, sample_num))
            batch_sample_ids = sample_ids[sample_start_id: sample_end_id]
            batch = self._get_batch(batch_sample_ids, train_data)
            batch = self._add_paddings(batch)

            loss, start_accuracy, end_accuracy, grad_norm, _, grad2norm = session.run(
                [self.loss, self.start_accuracy, self.end_accuracy, self.grad_norm, self.train_op,
                 self.grad2norm],
                feed_dict={self.contexts: batch[0], self.questions: batch[1],
                           self.context_lens: batch[2], self.question_lens: batch[3],
                           self.answer_start_ids: batch[4], self.answer_end_ids: batch[5]})

            _LOGGER.info('Processed batch {}/{}, loss={}, grad_norm={}'.format(
                batch_id + 1, batch_num, loss, grad_norm))
            losses.append(loss)
            start_accuracies.append(start_accuracy)
            end_accuracies.append(end_accuracy)

        return np.mean(losses), np.mean(start_accuracies), np.mean(end_accuracies)

    def _get_batch(self, sample_ids, data):
        if 'answer_start_ids' in data:
            batch = [[] for _ in range(6)]
        else:
            batch = [[] for _ in range(4)]

        for sample_id in sample_ids:
            batch[0].append(data['contexts'][sample_id])
            batch[1].append(data['questions'][sample_id])
            batch[2].append(data['context_lens'][sample_id])
            batch[3].append(data['question_lens'][sample_id])

            if 'answer_start_ids' in data:
                batch[4].append(data['answer_start_ids'][sample_id])
                batch[5].append(data['answer_end_ids'][sample_id])

        return batch

    def _add_paddings(self, batch):
        for data_id in range(len(batch)):
            data = copy(batch[data_id])
            if not isinstance(data[0], list):
                continue
            max_len = max(len(datum) for datum in data)
            for datum_id in range(len(data)):
                data[datum_id] = data[datum_id] + [0 for _ in range(max_len - len(data[datum_id]))]
            batch[data_id] = data
        return batch

    def search(self, start_probabilities, end_probabilities, max_span=15):
        context_len = start_probabilities.shape[1]
        predictions = []

        for sample_id in range(context_len):
            multiples = np.matmul(np.transpose(start_probabilities[sample_id: sample_id + 1]),
                                  end_probabilities[sample_id: sample_id+1])
            max_proba = 0.
            max_start_id = None
            max_end_id = None

            for start_id in range(context_len):
                for end_id in range(start_id, min(context_len, start_id + max_span)):
                    proba = multiples[start_id, end_id]
                    if proba > max_proba:
                        max_proba = proba
                        max_start_id = start_id
                        max_end_id = end_id

            predictions.append((max_start_id, max_end_id))

        return predictions

    def _build_graph(self):
        raise NotImplementedError()


class MatchLstmAnswerPointerModel(BaseQAModel):

    def _build_graph(self):
        with tf.variable_scope('inputs'):
            self.contexts = tf.placeholder(dtype=tf.int32, shape=(None, None), name='contexts')
            self.questions = tf.placeholder(dtype=tf.int32, shape=(None, None), name='questions')
            self.context_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name='context_lens')
            self.question_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name='question_lens')
            self.answer_start_ids = tf.placeholder(
                dtype=tf.int32, shape=(None,), name='answer_start_ids')
            self.answer_end_ids = tf.placeholder(
                dtype=tf.int32, shape=(None,), name='answer_end_ids')

        with tf.variable_scope('embedding'):
            self.embeddings = tf.Variable(
                np.load(self.params.embed_path)['glove'], trainable=False, name='embeddings')
            contexts = tf.cast(tf.nn.embedding_lookup(self.embeddings, self.contexts), tf.float32)
            questions = tf.cast(tf.nn.embedding_lookup(self.embeddings, self.questions), tf.float32)

        with tf.variable_scope('preprocessing'):
            # preprocess contexts
            fw_context_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            bw_context_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            contexts, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_context_cell, bw_context_cell, contexts, sequence_length=self.context_lens,
                dtype=tf.float32, scope='context/bidirectional_rnn')
            contexts = tf.concat(contexts, axis=-1)

            # preprocess questions
            fw_question_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            bw_question_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            questions, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_question_cell, bw_question_cell, questions, sequence_length=self.question_lens,
                dtype=tf.float32, scope='question/bidirectional_rnn')
            questions = tf.concat(questions, axis=-1)

        with tf.variable_scope('match-lstm'):
            # attention mechanism
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.params.state_size, memory=questions,
                memory_sequence_length=self.question_lens)

            # forward decoder
            fw_decoder_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            fw_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=fw_decoder_cell, attention_mechanism=attention_mechanism,
                output_attention=False)

            # backward decoder
            bw_decoder_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            bw_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=bw_decoder_cell, attention_mechanism=attention_mechanism,
                output_attention=False)

            # decoding with attention
            contexts, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_decoder_cell, bw_decoder_cell, contexts, sequence_length=self.context_lens,
                dtype=tf.float32, scope='bidirectional_rnn')

            contexts = tf.concat(contexts, axis=-1)

        with tf.variable_scope('answer-pointer'):
            batch_size = tf.shape(contexts)[0]
            answer_attention = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.params.state_size, memory=contexts,
                memory_sequence_length=self.context_lens)
            answer_decoder_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            cell_input_fn = lambda input_, attention: attention  # ignore the inputs
            answer_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=answer_decoder_cell, attention_mechanism=answer_attention,
                cell_input_fn=cell_input_fn, alignment_history=True)

            # we ignore the inputs
            dummy_inputs = [tf.zeros(shape=(batch_size,)), tf.zeros(shape=(batch_size,))]
            _, states = tf.nn.static_rnn(answer_decoder_cell, dummy_inputs, dtype=tf.float32)
            self.probabilities = states.alignment_history.stack()

        with tf.variable_scope('loss'):
            epsilon = 1e-8
            logits = tf.transpose(tf.log(self.probabilities + epsilon), (1, 0, 2))
            labels = tf.stack((self.answer_start_ids, self.answer_end_ids), axis=1)
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            grads, _ = list(zip(*grads_and_vars))
            self.train_op = optimizer.apply_gradients(grads_and_vars)
            self.grad_norm = tf.global_norm(grads)
            self.grad2norm = {}
            for grad, var in grads_and_vars:
                self.grad2norm[var.name] = tf.norm(grad)

        with tf.variable_scope('prediction'):
            self.start_probabilities = tf.gather(self.probabilities, 0)
            self.end_probabilities = tf.gather(self.probabilities, 1)
            self.start_predictions = tf.cast(tf.argmax(self.start_probabilities, axis=1), tf.int32)
            self.end_predictions = tf.cast(tf.argmax(self.end_probabilities, axis=1), tf.int32)
            self.start_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.start_predictions, self.answer_start_ids), dtype=tf.float32))
            self.end_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.end_predictions, self.answer_end_ids), dtype=tf.float32))

        # create summary
        for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            tf.summary.histogram(variable.name, variable)
        self.variable_summary = tf.summary.merge_all()

        self.saver = tf.train.Saver()


class BiLstmModel(BaseQAModel):

    def _build_graph(self):
        with tf.variable_scope('inputs'):
            self.contexts = tf.placeholder(dtype=tf.int32, shape=(None, None), name='contexts')
            self.questions = tf.placeholder(dtype=tf.int32, shape=(None, None), name='questions')
            self.context_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name='context_lens')
            self.question_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name='question_lens')
            self.answer_start_ids = tf.placeholder(
                dtype=tf.int32, shape=(None,), name='answer_start_ids')
            self.answer_end_ids = tf.placeholder(
                dtype=tf.int32, shape=(None,), name='answer_end_ids')

        with tf.variable_scope('embedding'):
            self.embeddings = tf.Variable(
                np.load(self.params.embed_path)['glove'], trainable=False, name='embeddings')
            contexts = tf.cast(tf.nn.embedding_lookup(self.embeddings, self.contexts), tf.float32)
            questions = tf.cast(tf.nn.embedding_lookup(self.embeddings, self.questions), tf.float32)

        with tf.variable_scope('preprocessing'):
            # preprocess contexts
            fw_context_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            bw_context_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            contexts, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_context_cell, bw_context_cell, contexts, sequence_length=self.context_lens,
                dtype=tf.float32, scope='context/bidirectional_rnn')
            contexts = tf.concat(contexts, axis=-1)

            # preprocess questions
            fw_question_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            bw_question_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            questions, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_question_cell, bw_question_cell, questions, sequence_length=self.question_lens,
                dtype=tf.float32, scope='question/bidirectional_rnn')
            questions = tf.concat(questions, axis=-1)
            question = self._mean_pool(questions, self.question_lens)

        with tf.variable_scope('fully-connected-layer'):
            context_shape = tf.shape(contexts)
            batch_size = context_shape[0]
            max_context_len = context_shape[1]

            questions = tf.tile(tf.expand_dims(question, axis=1), multiples=(1, max_context_len, 1))
            contexts_questions = tf.concat((contexts, questions), axis=-1)

            W_start = tf.get_variable(
                'W_start', shape=(self.params.state_size * 4, 1), dtype=tf.float32,
                initializer=xavier_initializer())
            b_start = tf.get_variable(
                'b_start', shape=(1,), dtype=tf.float32, initializer=tf.zeros_initializer())
            W_end = tf.get_variable(
                'W_end', shape=(self.params.state_size * 4, 1), dtype=tf.float32,
                initializer=xavier_initializer())
            b_end = tf.get_variable(
                'b_end', shape=(1,), dtype=tf.float32, initializer=tf.zeros_initializer())

            start_logits = tf.reshape(tf.matmul(
                tf.reshape(contexts_questions, (batch_size * max_context_len, -1)),
                W_start) + b_start, (batch_size, max_context_len, -1))
            start_logits = tf.squeeze(start_logits, axis=-1)
            end_logits = tf.reshape(tf.matmul(
                tf.reshape(contexts_questions, (batch_size * max_context_len, -1)),
                W_end) + b_end, (batch_size, max_context_len, -1))
            end_logits = tf.squeeze(end_logits, axis=-1)

        with tf.variable_scope('loss'):
            weights = tf.sequence_mask(self.context_lens, dtype=tf.float32)
            start_logits = tf.where(
                tf.cast(weights, tf.bool), start_logits,
                tf.multiply(tf.subtract(tf.ones(tf.shape(weights)), weights), -1e9))
            end_logits = tf.where(
                tf.cast(weights, tf.bool), end_logits,
                tf.multiply(tf.subtract(tf.ones(tf.shape(weights)), weights), -1e9))

            # start_labels = tf.one_hot(self.answer_start_ids, depth=max_context_len, dtype=tf.int32)
            # end_labels = tf.one_hot(self.answer_end_ids, depth=max_context_len, dtype=tf.int32)

            start_loss = tf.losses.sparse_softmax_cross_entropy(self.answer_start_ids, start_logits)
            end_loss = tf.losses.sparse_softmax_cross_entropy(self.answer_end_ids, end_logits)
            self.loss = start_loss + end_loss

        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            grads, _ = list(zip(*grads_and_vars))
            self.train_op = optimizer.apply_gradients(grads_and_vars)
            self.grad_norm = tf.global_norm(grads)
            self.grad2norm = {}
            for grad, var in grads_and_vars:
                self.grad2norm[var.name] = tf.norm(grad)

        with tf.variable_scope('prediction'):
            self.start_probabilities = tf.nn.softmax(start_logits)
            self.end_probabilities = tf.nn.softmax(end_logits)
            self.start_predictions = tf.cast(tf.argmax(self.start_probabilities, axis=1), tf.int32)
            self.end_predictions = tf.cast(tf.argmax(self.end_probabilities, axis=1), tf.int32)
            self.start_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.start_predictions, self.answer_start_ids), dtype=tf.float32))
            self.end_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.end_predictions, self.answer_end_ids), dtype=tf.float32))

        # create summary
        for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            tf.summary.histogram(variable.name, variable)
        self.variable_summary = tf.summary.merge_all()

        self.saver = tf.train.Saver()

    def _mean_pool(self, outputs, seq_lens):
        """
        Perform mean-pooling over time

        :param outputs: hidden states of all the time steps
        :param seq_lens: list of sequence length excluding paddings
        :return: mean values of each hidden state dimension over time
        """
        max_len = tf.reduce_max(seq_lens)

        # take mean over time
        outputs = tf.reduce_mean(outputs, axis=1)

        # In order to avoid 0 paddings affect the mean, multiply by `n / m` where `n` is
        # `max_len` and `m` is `seq_lens`
        return tf.transpose(
            tf.div(
                tf.transpose(
                    tf.multiply(outputs, tf.cast(max_len, tf.float32))),
                tf.cast(seq_lens, tf.float32)))


class LuongAttention(BaseQAModel):

    def _build_graph(self):

        with tf.variable_scope('inputs'):
            self.contexts = tf.placeholder(dtype=tf.int32, shape=(None, None), name='contexts')
            self.questions = tf.placeholder(dtype=tf.int32, shape=(None, None), name='questions')
            self.context_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name='context_lens')
            self.question_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name='question_lens')
            self.answer_start_ids = tf.placeholder(
                dtype=tf.int32, shape=(None,), name='answer_start_ids')
            self.answer_end_ids = tf.placeholder(
                dtype=tf.int32, shape=(None,), name='answer_end_ids')

        with tf.variable_scope('embedding'):
            self.embeddings = tf.Variable(
                np.load(self.params.embed_path)['glove'], trainable=False, name='embeddings')
            contexts = tf.cast(tf.nn.embedding_lookup(self.embeddings, self.contexts), tf.float32)
            questions = tf.cast(tf.nn.embedding_lookup(self.embeddings, self.questions), tf.float32)

        with tf.variable_scope('encoder'):
            # preprocess questions
            fw_question_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            bw_question_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            questions, (question_state_fw, question_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                fw_question_cell, bw_question_cell, questions, sequence_length=self.question_lens,
                dtype=tf.float32, scope='question/bidirectional_rnn')
            questions = tf.concat(questions, axis=-1)

            # preprocess contexts
            fw_context_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            bw_context_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            contexts, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_context_cell, bw_context_cell, contexts, initial_state_fw=question_state_fw,
                initial_state_bw=question_state_bw, sequence_length=self.context_lens,
                dtype=tf.float32, scope='context/bidirectional_rnn')
            contexts = tf.concat(contexts, axis=-1)

        with tf.variable_scope('attention'):
            context_shape = tf.shape(contexts)
            batch_size = context_shape[0]
            max_context_len = context_shape[1]
            max_question_len = tf.shape(questions)[1]

            interaction_weights = tf.get_variable(
                'interaction_weights',
                shape=(self.params.state_size * 2, self.params.state_size * 2),
                initializer=xavier_initializer(), dtype=tf.float32)

            weighted_contexts = tf.reshape(tf.matmul(
                tf.reshape(contexts, (batch_size * max_context_len, -1)), interaction_weights),
                (batch_size, max_context_len, -1))
            alignment_scores = tf.matmul(weighted_contexts, tf.transpose(questions, (0, 2, 1)))

            context_mask = tf.sequence_mask(self.context_lens, dtype=tf.float32)
            question_mask = tf.sequence_mask(self.question_lens, dtype=tf.float32)
            alignment_mask = tf.multiply(
                tf.tile(tf.expand_dims(context_mask, 2), [1, 1, max_question_len]),
                tf.tile(tf.expand_dims(question_mask, 1), [1, max_context_len, 1]))
            alignment_scores = tf.where(
                tf.cast(alignment_mask, tf.bool), alignment_scores,
                tf.multiply(tf.subtract(tf.ones(tf.shape(alignment_mask)), alignment_scores), -1e9))

            alignment_weights = tf.nn.softmax(alignment_scores)
            context_aware = tf.matmul(alignment_weights, questions)
            concat_hidden = tf.concat((context_aware, contexts), axis=2)

            W_attention = tf.get_variable(
                'W_attention', shape=(self.params.state_size * 4, self.params.state_size * 2),
                initializer=xavier_initializer(), dtype=tf.float32)
            attentions = tf.nn.tanh(
                tf.reshape(tf.matmul(tf.reshape(
                    concat_hidden, (batch_size * max_context_len, -1)), W_attention),
                    (batch_size, max_context_len, self.params.state_size * 2)))

        with tf.variable_scope('decoder1'):
            decode1_fw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            decode1_bw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            activations1, _ = tf.nn.bidirectional_dynamic_rnn(
                decode1_fw_cell, decode1_bw_cell, attentions, sequence_length=self.context_lens,
                dtype=tf.float32, scope='decoder1/bidirectional_rnn')
            activations1 = tf.concat(activations1, axis=-1)

        with tf.variable_scope('decoder2'):
            decode2_fw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            decode2_bw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.state_size)
            activations2, _ = tf.nn.bidirectional_dynamic_rnn(
                decode2_fw_cell, decode2_bw_cell, activations1, sequence_length=self.context_lens,
                dtype=tf.float32, scope='decoder2/bidirectional_rnn')
            activations2 = tf.concat(activations2, axis=-1)

        with tf.variable_scope('logits'):
            W_start = tf.get_variable('W_start', shape=(self.params.state_size * 2, 1),
                                      initializer=xavier_initializer(), dtype=tf.float32)
            start_logits = tf.reshape(tf.matmul(tf.reshape(
                activations2, (batch_size * max_context_len, -1)), W_start),
                (batch_size, max_context_len, 1))
            start_logits = tf.squeeze(start_logits, axis=-1)
            start_logits = tf.where(
                tf.cast(context_mask, tf.bool), start_logits,
                tf.multiply(tf.subtract(tf.ones(tf.shape(context_mask)), start_logits), -1e9))

            W_end = tf.get_variable('W_end', shape=(self.params.state_size * 2, 1),
                                    initializer=xavier_initializer(), dtype=tf.float32)
            end_logits = tf.reshape(tf.matmul(tf.reshape(
                activations2, (batch_size * max_context_len, -1)), W_end),
                (batch_size, max_context_len, 1))
            end_logits = tf.squeeze(end_logits, axis=-1)
            end_logits = tf.where(
                tf.cast(context_mask, tf.bool), end_logits,
                tf.multiply(tf.subtract(tf.ones(tf.shape(context_mask)), end_logits), -1e9))

        with tf.variable_scope('loss'):
            start_loss = tf.losses.sparse_softmax_cross_entropy(self.answer_start_ids, start_logits)
            end_loss = tf.losses.sparse_softmax_cross_entropy(self.answer_end_ids, end_logits)
            self.loss = start_loss + end_loss

        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            grads, _ = list(zip(*grads_and_vars))
            self.train_op = optimizer.apply_gradients(grads_and_vars)
            self.grad_norm = tf.global_norm(grads)
            self.grad2norm = {}
            for grad, var in grads_and_vars:
                self.grad2norm[var.name] = tf.norm(grad)

        with tf.variable_scope('prediction'):
            self.start_probabilities = tf.nn.softmax(start_logits)
            self.end_probabilities = tf.nn.softmax(end_logits)
            self.start_predictions = tf.cast(tf.argmax(self.start_probabilities, axis=1), tf.int32)
            self.end_predictions = tf.cast(tf.argmax(self.end_probabilities, axis=1), tf.int32)
            self.start_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.start_predictions, self.answer_start_ids), dtype=tf.float32))
            self.end_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.end_predictions, self.answer_end_ids), dtype=tf.float32))

        # create summary
        for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            tf.summary.histogram(variable.name, variable)
        self.variable_summary = tf.summary.merge_all()

        self.saver = tf.train.Saver()
