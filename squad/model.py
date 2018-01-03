# -*- coding: UTF-8 -*-
"""Define QA models here.
"""
import logging
import os
from copy import copy
from random import shuffle

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

__author__ = 'Kensuke Muraki'

_LOGGER = logging.getLogger(__name__)


class HParams(object):

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class QAModel(object):

    def __init__(self, params):
        self.params = params
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_graph()

    def fit(self, train_data, valid_data, train_dir, batch_size=10, epochs=10):
        sample_num = len(train_data['contexts'])
        sample_ids = range(len(train_data['contexts']))

        with tf.Session(graph=self.graph) as session:
            session.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(
                os.path.join(train_dir, 'tensorboard.log'), session.graph)

            for epoch_id in range(epochs):
                shuffle(sample_ids)
                _LOGGER.info('--- Start epoch_id={} ---'.format(epoch_id))
                train_loss, train_start_accuracy, train_end_accuracy = self._train_epoch(
                    train_data, batch_size, sample_ids, sample_num, session)

                _LOGGER.info('Finished training epoch_id={}.\n'.format(epoch_id))
                _LOGGER.info('Train stats: loss={}, start_accuracy={}, end_accuracy={}\n'.format(
                    train_loss, train_start_accuracy, train_end_accuracy))

                valid_loss, valid_exact_match, valid_f1 = self._validate(valid_data, session)

                _LOGGER.info('Finished validation.')
                _LOGGER.info('Validation stats: loss={}, exact_match={}, f1={}\n'.format(
                    valid_loss, valid_exact_match, valid_f1))

                self._add_summary(epoch_id, session, summary_writer, train_end_accuracy, train_loss,
                                  train_start_accuracy, valid_exact_match, valid_f1, valid_loss)

                # save the model
                _LOGGER.info('Saving the model...')
                self.saver.save(session, os.path.join(train_dir, 'model.ckpt'))

    def _add_summary(self, epoch_id, session, summary_writer, train_end_accuracy, train_loss,
                     train_start_accuracy, valid_exact_match, valid_f1, valid_loss):
        # add histogram summary of variables
        summary_writer.add_summary(session.run(self.variable_summary), epoch_id)

        metric_summary = tf.Summary()

        # add train summary
        metric_summary.value.add(tag='train_loss', simple_value=train_loss)
        metric_summary.value.add(tag='train_start_accuracy', simple_value=train_start_accuracy)
        metric_summary.value.add(tag='train_end_accuracy', simple_value=train_end_accuracy)

        # add validation summary
        metric_summary.value.add(tag='valid_loss', simple_value=valid_loss)
        metric_summary.value.add(tag='valid_exact_match', simple_value=valid_exact_match)
        metric_summary.value.add(tag='valid_f1', simple_value=valid_f1)

        summary_writer.add_summary(metric_summary, global_step=epoch_id)

    def _validate(self, valid_data, session):
        batch = self._get_batch(range(len(valid_data['contexts'])), valid_data)
        batch = self._add_paddings(batch)
        loss, start_probabilities, end_probabilities, _ = session.run(
            [self.loss, self.start_probabilities, self.end_probabilities, self.optimizer],
            feed_dict={self.contexts: batch[0], self.questions: batch[1],
                       self.context_lens: batch[2], self.question_lens: batch[3],
                       self.answer_start_ids: batch[4], self.answer_end_ids: batch[5]})
        predictions = self.search(start_probabilities, end_probabilities)
        exact_match, f1 = self.compute_metrics(predictions, batch[4], batch[5])
        return loss, exact_match, f1

    def compute_metrics(self, predictions, answer_start_ids, answer_end_ids):
        exact_matches = []
        f1 = []

        for prediction, answer in zip(predictions, zip(answer_start_ids, answer_end_ids)):
            exact_matches.append(int(prediction == answer))
            prediction = set(range(prediction[0], prediction[1] + 1))
            answer = set(range(answer[0], answer[1] + 1))
            true_positive = float(len(prediction.intersection(answer)))
            false_positive = float(len(prediction.difference(answer)))
            false_negative = float(len(answer.difference(prediction)))
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)

            if precision == 0. and recall == 0:
                f1.append(0.)
            else:
                f1.append(2 * precision * recall / (precision + recall))

        return np.mean(exact_matches), np.mean(f1)

    def _train_epoch(self, train_data, batch_size, sample_ids, sample_num, session):
        losses = []
        start_accuracies = []
        end_accuracies = []
        batch_num = sample_num // batch_size

        for batch_id in range(batch_num):
            batch_sample_ids = sample_ids[batch_id * batch_size: (batch_id + 1) * batch_size]
            batch = self._get_batch(batch_sample_ids, train_data)
            batch = self._add_paddings(batch)

            loss, start_accuracy, end_accuracy, _ = session.run(
                [self.loss, self.start_accuracy, self.end_accuracy, self.optimizer],
                feed_dict={self.contexts: batch[0], self.questions: batch[1],
                           self.context_lens: batch[2], self.question_lens: batch[3],
                           self.answer_start_ids: batch[4], self.answer_end_ids: batch[5]})

            _LOGGER.info('Processed batch {}/{}, loss={}'.format(batch_id + 1, batch_num, loss))
            losses.append(loss)
            start_accuracies.append(start_accuracy)
            end_accuracies.append(end_accuracy)

        return np.mean(losses), np.mean(start_accuracies), np.mean(end_accuracies)

    def _get_batch(self, sample_ids, train_data):
        batch = [[] for _ in range(6)]
        for sample_id in sample_ids:
            batch[0].append(train_data['contexts'][sample_id])
            batch[1].append(train_data['questions'][sample_id])
            batch[2].append(train_data['context_lens'][sample_id])
            batch[3].append(train_data['question_lens'][sample_id])
            batch[4].append(train_data['answer_start_ids'][sample_id])
            batch[5].append(train_data['answer_end_ids'][sample_id])
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
        context_len = len(start_probabilities)
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
            fw_context_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.params.state_size)
            bw_context_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.params.state_size)
            contexts, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_context_cell, bw_context_cell, contexts, sequence_length=self.context_lens,
                dtype=tf.float32, scope='context/bidirectional_rnn')
            contexts = tf.concat(contexts, axis=2)

            # preprocess questions
            fw_question_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.params.state_size)
            bw_question_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.params.state_size)
            questions, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_question_cell, bw_question_cell, questions, sequence_length=self.question_lens,
                dtype=tf.float32, scope='question/bidirectional_rnn')
            questions = tf.concat(questions, axis=2)

        with tf.variable_scope('match-lstm'):
            decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.params.state_size * 2)
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=self.params.state_size * 2, memory=questions,
                memory_sequence_length=self.question_lens)
            fw_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=decoder_cell, attention_mechanism=attention_mechanism,
                attention_layer_size=self.params.state_size * 2)
            bw_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=decoder_cell, attention_mechanism=attention_mechanism,
                attention_layer_size=self.params.state_size * 2)
            contexts, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_decoder_cell, bw_decoder_cell, contexts, sequence_length=self.context_lens,
                dtype=tf.float32)
            contexts = tf.concat(contexts, axis=2)

        with tf.variable_scope('answer-pointer'):
            # compute shapes
            context_shape = tf.shape(contexts)
            batch_size = context_shape[0]
            max_context_len = context_shape[1]

            # for answer start pointer
            w_start = tf.get_variable('W_start', shape=(self.params.state_size * 4, 1),
                                      initializer=xavier_initializer(), dtype=tf.float32)
            b_start = tf.get_variable('b_start', shape=(1,), initializer=tf.zeros_initializer(),
                                      dtype=tf.float32)
            start_logits = tf.reshape(
                tf.matmul(tf.reshape(contexts, (batch_size * max_context_len, -1)),
                          w_start) + b_start, (batch_size, max_context_len, 1))
            start_logits = tf.squeeze(start_logits, axis=2)

            # for answer end pointer
            w_end = tf.get_variable('W_end', shape=(self.params.state_size * 4, 1),
                                    initializer=xavier_initializer(), dtype=tf.float32)
            b_end = tf.get_variable('b_end', shape=(1,), initializer=tf.zeros_initializer(),
                                    dtype=tf.float32)
            end_logits = tf.reshape(
                tf.matmul(tf.reshape(contexts, (batch_size * max_context_len, -1)), w_end) + b_end,
                (batch_size, max_context_len, 1))
            end_logits = tf.squeeze(end_logits, axis=2)

        with tf.variable_scope('loss'):
            # add large values to valid logits so that softmax are near accurate even with paddings
            start_logits = start_logits + tf.multiply(
                tf.sequence_mask(self.context_lens, dtype=tf.float32), self.params.large_value)
            end_logits = end_logits + tf.multiply(
                tf.sequence_mask(self.context_lens, dtype=tf.float32), self.params.large_value)
            start_loss = tf.losses.sparse_softmax_cross_entropy(self.answer_start_ids, start_logits)
            end_loss = tf.losses.sparse_softmax_cross_entropy(self.answer_end_ids, end_logits)
            self.loss = start_loss + end_loss

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.params.learning_rate).minimize(self.loss)

        with tf.variable_scope('prediction'):
            self.start_probabilities = tf.nn.softmax(start_logits)
            self.start_predictions = tf.cast(tf.argmax(self.start_probabilities, axis=1), tf.int32)
            self.end_probabilities = tf.nn.softmax(end_logits)
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
