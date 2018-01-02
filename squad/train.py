# -*- coding: UTF-8 -*-
"""Train a QA model.
"""
import os
import logging
from collections import namedtuple
from random import shuffle
from copy import copy

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer

__author__ = 'Kensuke Muraki'

_LOGGER = logging.getLogger(__name__)

LARGE_VALUE = 100
TRAIN_DIR = 'train'


def load_data(data_dir, debug=False):
    data = {}
    prefixes = ['train', 'val']
    fields = ['ids.context', 'ids.question', 'span']

    for prefix in prefixes:
        data[prefix] = {}
        for field in fields:
            data[prefix][field] = []
            with open(os.path.join(data_dir, '{}.{}'.format(prefix, field)), 'r') as file_:
                for line in file_:
                    if debug and len(data[prefix][field]) >= 50:
                        break
                    vals = [int(val) for val in line.strip().split()]
                    data[prefix][field].append(vals)

    return data


class QAModel(object):

    def __init__(self, params):
        self.params = params
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_graph()

    def fit(self, train_data, valid_data, batch_size=10, epochs=10):
        sample_num = len(train_data['contexts'])
        sample_ids = range(len(train_data['contexts']))

        with tf.Session(graph=self.graph) as session:
            session.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(
                os.path.join(TRAIN_DIR, 'tensorboard.log'), session.graph)

            for epoch_id in range(epochs):
                shuffle(sample_ids)
                print('epoch_id={}'.format(epoch_id))
                train_loss, train_start_accuracy, train_end_accuracy = self._train_epoch(
                    train_data, batch_size, sample_ids, sample_num, session)

                print('Finished training epoch_id={}.'.format(epoch_id))
                print('Train stats: loss={}, start_accuracy={}, end_accuracy={}'.format(
                    train_loss, train_start_accuracy, train_end_accuracy))

                # add train summary
                summary_writer.add_summary(session.run(self.variable_summary), epoch_id)
                summary_writer.add_summary(
                    tf.summary.scalar('train_loss', train_loss).eval(), epoch_id)
                summary_writer.add_summary(
                    tf.summary.scalar(
                        'train_start_accuracy', train_start_accuracy).eval(), epoch_id)
                summary_writer.add_summary(
                    tf.summary.scalar('train_end_accuracy', train_end_accuracy).eval(), epoch_id)

                valid_loss, valid_start_accuracy, valid_end_accuracy = self._validate(
                    valid_data, session)

                print('Finished validation')
                print('loss={}, start_accuracy={}, end_accuracy={}'.format(
                    valid_loss, valid_start_accuracy, valid_end_accuracy))

                # add validation summary
                summary_writer.add_summary(
                    tf.summary.scalar('valid_loss', valid_loss).eval(), epoch_id)
                summary_writer.add_summary(
                    tf.summary.scalar(
                        'valid_start_accuracy', valid_start_accuracy).eval(), epoch_id)
                summary_writer.add_summary(
                    tf.summary.scalar('valid_end_accuracy', valid_end_accuracy).eval(), epoch_id)

                # save the model
                print('Saving the model...')
                self.saver.save(session, os.path.join(TRAIN_DIR, 'model.ckpt'))

    def _validate(self, valid_data, session):
        batch = self._get_batch(range(len(valid_data['contexts'])), valid_data)
        batch = self._add_paddings(batch)
        loss, start_accuracy, end_accuracy, _ = session.run(
            [self.loss, self.start_accuracy, self.end_accuracy, self.optimizer],
            feed_dict={self.contexts: batch[0], self.questions: batch[1],
                       self.context_lens: batch[2], self.question_lens: batch[3],
                       self.answer_start_ids: batch[4], self.answer_end_ids: batch[5]})
        return loss, start_accuracy, end_accuracy

    def _train_epoch(self, train_data, batch_size, sample_ids, sample_num, session):
        losses = []
        start_accuracies = []
        end_accuracies = []

        for batch_id in range(sample_num // batch_size):
            batch_sample_ids = sample_ids[batch_id * batch_size: (batch_id + 1) * batch_size]
            batch = self._get_batch(batch_sample_ids, train_data)
            batch = self._add_paddings(batch)
            loss, start_accuracy, end_accuracy, _ = session.run(
                [self.loss, self.start_accuracy, self.end_accuracy, self.optimizer],
                feed_dict={self.contexts: batch[0], self.questions: batch[1],
                           self.context_lens: batch[2], self.question_lens: batch[3],
                           self.answer_start_ids: batch[4], self.answer_end_ids: batch[5]})
            print('batch_id={}, loss={}'.format(batch_id, loss))
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

        with tf.name_scope('loss'):
            # add large values to valid logits so that softmax are near accurate even with paddings
            start_logits = start_logits + tf.multiply(
                tf.sequence_mask(self.context_lens, dtype=tf.float32), LARGE_VALUE)
            end_logits = end_logits + tf.multiply(
                tf.sequence_mask(self.context_lens, dtype=tf.float32), LARGE_VALUE)
            start_loss = tf.losses.sparse_softmax_cross_entropy(self.answer_start_ids, start_logits)
            end_loss = tf.losses.sparse_softmax_cross_entropy(self.answer_end_ids, end_logits)
            self.loss = start_loss + end_loss

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.params.learning_rate).minimize(self.loss)

        with tf.name_scope('prediction'):
            self.start_probabilities = tf.nn.softmax(start_logits)
            self.start_predictions = tf.argmax(self.start_probabilities, axis=1)
            self.end_probabilities = tf.nn.softmax(end_logits)
            self.end_predictions = tf.argmax(self.end_probabilities, axis=1)
            self.start_accuracy = tf.reduce_mean(
                tf.cast(self.start_predictions == self.answer_start_ids, dtype=tf.float32))
            self.end_accuracy = tf.reduce_mean(
                tf.cast(self.end_predictions == self.answer_end_ids, dtype=tf.float32))

        with tf.name_scope('summary'):
            for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                tf.summary.histogram(variable.name, variable)
            self.variable_summary = tf.summary.merge_all()

        self.saver = tf.train.Saver()

        # TODO: implement decoding
        # TODO: implement metrics computation (accuracy (for both start/end), exact match, f1)


# TODO: these numbers can be and should be tuned (define them as hyper-parameters)
def filter_data(data, max_context_len=300, max_question_len=25):
    idx = 0
    while idx < len(data['ids.context']):
        if len(data['ids.context'][idx]) > max_context_len or \
                len(data['ids.question'][idx]) > max_question_len:
            del data['ids.context'][idx]
            del data['ids.question'][idx]
            del data['span'][idx]
        else:
            idx += 1
    return data


def preprocess_data(data, filter=True):
    if filter:
        data = filter_data(data)
    context_lens = [len(context) for context in data['ids.context']]
    question_lens = [len(question) for question in data['ids.question']]
    answer_start_ids, answer_end_ids = zip(*data['span'])
    data = {
        'contexts': data['ids.context'],
        'questions': data['ids.question'],
        'context_lens': context_lens,
        'question_lens': question_lens,
        'answer_start_ids': answer_start_ids,
        'answer_end_ids': answer_end_ids,
    }
    return data


def main():
    data = load_data('data/squad', debug=True)
    train_data = preprocess_data(data['train'])
    valid_data = preprocess_data(data['val'], filter=False)
    params = {
        'learning_rate': 0.001,
        'batch_size': 10,
        'epochs': 10,
        'state_size': 200,
        'embed_path': 'data/squad/glove.trimmed.100.npz',
    }
    Params = namedtuple(
        'params', params.keys())
    params = Params(**params)
    qa_model = QAModel(params)
    qa_model.fit(
        train_data, valid_data, epochs=params.epochs, batch_size=params.batch_size)


if __name__ == '__main__':
    main()
