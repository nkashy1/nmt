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

"""Data loader tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import eeap_data
import tensorflow as tf


DATA_DIR = './testdata-eeap'


class LabelledDataTests(tf.test.TestCase):
  def setUp(self):
    self.source_file = '{}/source.txt'.format(DATA_DIR)
    self.target_file = '{}/target.txt'.format(DATA_DIR)

    with open(self.source_file, 'r') as src:
      self.source = src.read().strip().split('\n')

    with open(self.target_file, 'r') as tgt:
      self.target = tgt.read().strip().split('\n')

  def test_generate_labelled_input_fn_1(self):
    batch_size = 1

    input_fn = eeap_data.generate_labelled_input_fn(
      self.source_file,
      self.target_file,
      batch_size=batch_size)

    features, labels = input_fn()

    with tf.Session() as sess:
      result = sess.run({'features': features, 'labels': labels})

    self.assertListEqual(list(result['features']), self.source[:batch_size])
    self.assertListEqual(list(result['labels']), self.target[:batch_size])

  def test_generate_labelled_input_fn_2(self):
    batch_size = 2

    input_fn = eeap_data.generate_labelled_input_fn(
      self.source_file,
      self.target_file,
      batch_size=batch_size)

    features, labels = input_fn()

    with tf.Session() as sess:
      result = sess.run({'features': features, 'labels': labels})

    self.assertListEqual(list(result['features']), self.source[:batch_size])
    self.assertListEqual(list(result['labels']), self.target[:batch_size])

  def test_generate_labelled_input_fn_3(self):
    batch_size = 3

    input_fn = eeap_data.generate_labelled_input_fn(
      self.source_file,
      self.target_file,
      batch_size)

    features, labels = input_fn()

    with tf.Session() as sess:
      result = sess.run({'features': features, 'labels': labels})

    self.assertListEqual(list(result['features']), self.source[:batch_size])
    self.assertListEqual(list(result['labels']), self.target[:batch_size])

  def test_dict_wrapper(self):
    batch_size = 2

    input_fn = eeap_data.dict_wrapper(eeap_data.generate_labelled_input_fn(
      self.source_file,
      self.target_file,
      batch_size))

    features, labels = input_fn()

    with tf.Session() as sess:
      result = sess.run({'features': features, 'labels': labels})

    self.assertIn(eeap_data.DEFAULT_KEY, result['features'])
    self.assertIn(eeap_data.DEFAULT_KEY, result['labels'])

    self.assertListEqual(list(result['features'][eeap_data.DEFAULT_KEY]),
                         self.source[:batch_size])

    self.assertListEqual(list(result['labels'][eeap_data.DEFAULT_KEY]),
                         self.target[:batch_size])


if __name__ == '__main__':
  tf.test.main()
