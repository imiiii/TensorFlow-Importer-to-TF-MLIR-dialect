import iree.compiler.tf
import iree.runtime
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow.compat.v2 as tf
loaded_model = tf.saved_model.load('/home/imi/programming/saved_model/my_model/')
call = loaded_model.__call__.get_concrete_function(
         tf.TensorSpec(shape=(None,28, 28), dtype=tf.float32, name='flatten_2_input'))
signatures = {'predict': call}
tf.saved_model.save(loaded_model,
  '/home/imi/programming/saved_model/resaved/', signatures=signatures)
