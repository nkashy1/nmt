"""Export inference graph for serving"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from . import attention_model
from . import model_helper
from . import model as nmt_model
from . import gnmt_model
from . import inference

def export(ckpt, hparams, export_dir):
  if not hparams.attention:
    model_creator = nmt_model.Model
  elif hparams.attention_architecture == "standard":
    model_creator = attention_model.AttentionModel
  elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
    model_creator = gnmt_model.GNMTModel
  else:
    raise ValueError("Unknown model architecture")

  infer_model = model_helper.create_infer_model(model_creator, hparams)

  builder = tf.saved_model_builder.SavedModelBuilder(export_dir)

  with tf.Session(graph=infer_model.graph) as sess:
    loaded_infer_model = model_helper.load_model(
      infer_model.model,
      ckpt,
      sess
    )

    tags = [tf.saved_model.tag_constants.SERVING]

    signature_def = tf.saved_model.signature_def_utils.predict_signature_def(
      {'src': infer_model.src_placeholder}
    )

    builder.add_meta_graph_and_variables(
      sess,
      tags,
      {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          signature_def
      },
      clear_devices=True
    )

  return
