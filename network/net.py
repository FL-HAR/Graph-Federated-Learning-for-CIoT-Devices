import tensorflow as tf
import functools
def CNN():
  """The CNN model used in https://arxiv.org/abs/1602.05629."""
  data_format = 'channels_last'
  input_shape = [32, 32,1]

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu)

  model = tf.keras.models.Sequential([
      conv2d(filters=1, input_shape=input_shape),
      max_pool(),
      conv2d(filters=32),
      max_pool(),
      conv2d(filters=64),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation=tf.nn.relu),
      tf.keras.layers.Dense(10),
  ])

  return model