import tensorflow as tf

class MyCustomLoss(tf.keras.losses.Loss):

  def call(self, y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    loss_obj = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
    cat_loss = loss_obj(y_true, y_pred)

    loss_obj = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM)
    kl_loss = loss_obj(y_true, y_pred)

    return cat_loss + kl_loss