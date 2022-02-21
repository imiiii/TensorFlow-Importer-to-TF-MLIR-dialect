import iree.compiler.tf
import iree.runtime
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow.compat.v2 as tf
loaded_model = tf.saved_model.load('/home/imi/programming/resaved/')
print(list(loaded_model.signatures.keys()))

