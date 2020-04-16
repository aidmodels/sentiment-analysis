import tensorflow as tf

pre_model = tf.keras.models.load_model("pretrained/model.h5")
pre_model.save("pretrained/model")