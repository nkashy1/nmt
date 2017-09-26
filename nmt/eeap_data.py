# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data loaders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def generate_labelled_input_fn(source_file,
                               target_file,
                               batch_size,
                               epochs=None):
  def input_fn():
    source_data = tf.contrib.data.TextLineDataset([source_file])
    target_data = tf.contrib.data.TextLineDataset([target_file])

    zipped_data = tf.contrib.data.Dataset.zip((source_data, target_data))
    batched_data = zipped_data.batch(batch_size)
    labelled_data = batched_data.repeat(epochs)
    labelled_data_iterator = labelled_data.make_one_shot_iterator()

    features, labels = labelled_data_iterator.get_next()
    return features, labels

  return input_fn


DEFAULT_KEY = 'sentences'


def dict_wrapper(input_fn,
                 features_key=DEFAULT_KEY,
                 labels_key=DEFAULT_KEY):
  def wrapped_input_fn():
    features, labels = input_fn()
    return {features_key: features}, {labels_key: labels}

  return wrapped_input_fn
