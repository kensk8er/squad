# -*- coding: UTF-8 -*-
"""Train a QA model.

"""
import os
import logging

import tensorflow as tf

from squad.model import HParams, BiLstmModel, LuongAttention, MatchLstmAnswerPointerModel

__author__ = 'Kensuke Muraki'

tf.app.flags.DEFINE_float("learning_rate", 0.005, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.75, "Dropout keep probability rate.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory")
tf.app.flags.DEFINE_string(
    "train_dir", "train", "Training directory to save the model parameters./train).")
tf.app.flags.DEFINE_string(
    "load_train_dir", "train",
    "Training directory to load model parameters from to resume training.")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string(
    "vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string(
    "embed_path", "data/squad/glove.trimmed.100.npz", "Path to the trimmed GLoVe embedding")
tf.app.flags.DEFINE_integer(
    "max_context_len", 400,
    "Maximum context length in training (filter out examples with higher context length)")
tf.app.flags.DEFINE_integer(
    "max_question_len", 25,
    "Maximum question length in training (filter out examples with higher question length)")
tf.app.flags.DEFINE_integer("max_span", 20, "Maximum span length to search.")
tf.app.flags.DEFINE_string("architecture", "Luong", "Architecture of the model to train.")

FLAGS = tf.app.flags.FLAGS


def load_data(data_dir, sample_num=None):
    data = {}
    prefixes = ['train', 'dev']
    fields = ['ids.context', 'ids.question', 'span']

    for prefix in prefixes:
        data[prefix] = {}
        for field in fields:
            data[prefix][field] = []
            with open(os.path.join(data_dir, '{}.{}'.format(prefix, field)), 'r') as file_:
                for line in file_:
                    if sample_num and len(data[prefix][field]) >= sample_num:
                        break

                    if prefix == 'dev' and field == 'span':
                        vals = []
                        for span in line.strip().split('\t'):
                            vals.append([int(val) for val in span.split()])
                    else:
                        vals = [int(val) for val in line.strip().split()]

                    data[prefix][field].append(vals)

    return data


def filter_data(data, max_context_len, max_question_len):
    if max_context_len is None:
        max_context_len = float('inf')
    if max_question_len is None:
        max_question_len = float('inf')

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


def preprocess_data(data, mode, max_context_len=None, max_question_len=None):
    if max_context_len or max_question_len:
        data = filter_data(data, max_context_len, max_question_len)
    context_lens = [len(context) for context in data['ids.context']]
    question_lens = [len(question) for question in data['ids.question']]

    if mode == 'train':
        answer_start_ids, answer_end_ids = zip(*data['span'])
        return {
            'contexts': data['ids.context'],
            'questions': data['ids.question'],
            'context_lens': context_lens,
            'question_lens': question_lens,
            'answer_start_ids': answer_start_ids,
            'answer_end_ids': answer_end_ids,
        }
    else:
        return {
            'contexts': data['ids.context'],
            'questions': data['ids.question'],
            'context_lens': context_lens,
            'question_lens': question_lens,
            'spans': data['span'],
        }


def main(_):
    data = load_data(FLAGS.data_dir)
    train_data = preprocess_data(data['train'], 'train', max_context_len=FLAGS.max_context_len, 
                                 max_question_len=FLAGS.max_question_len)
    valid_data = preprocess_data(data['dev'], 'dev')
    hyper_parameters = HParams(learning_rate=FLAGS.learning_rate, state_size=FLAGS.state_size,
                               embed_path=FLAGS.embed_path, dropout=FLAGS.dropout,
                               max_span=FLAGS.max_span)

    architectures = {
        'Luong': LuongAttention,
        'BiLSTM': BiLstmModel,
        'Match': MatchLstmAnswerPointerModel,
    }
    qa_model = architectures[FLAGS.architecture](hyper_parameters)
    qa_model.fit(
        train_data, valid_data, train_dir=FLAGS.train_dir, epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
