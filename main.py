import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from cvae import CVAE

flags = tf.flags
flags.DEFINE_integer("latent_size", 200, "Number of latent variables.")
flags.DEFINE_integer("epochs", 100, "Maximum number of epochs.")
flags.DEFINE_integer("batch_size", 128, "Mini-batch size for data subsampling.")
flags.DEFINE_string("device", "/gpu:0", "Compute device.")
flags.DEFINE_string("dataset_dir", "", "Directory of the input dataset.")
flags.DEFINE_boolean("allow_soft_placement", True, "Soft device placement.")
flags.DEFINE_float("device_percentage", "1.0", "Amount of memory to use on device.")
FLAGS = flags.FLAGS

def normalize(arr):
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-9)

def build_cvae(sess, input_files, input_shape, latent_size, batch_size, epochs=100):
    cvae = CVAE(sess, input_shape, batch_size, latent_size=latent_size)
    model_filename = "models/%s.cpkt" % cvae.get_name()
    if os.path.isfile(model_filename):
        cvae.load(sess, model_filename)
    else:
        sess.run(tf.initialize_all_variables())
        cvae.train(sess, input_files, batch_size, display_step=1, training_epochs=epochs)
        cvae.save(sess, model_filename)

    return cvae

def main():
    input_files = glob('{}/*.nii.gz'.format(FLAGS.dataset_dir))
    input_shape = [91, 109, 91]

    # model storage
    if not os.path.exists('models'):
        os.makedirs('models')

    with tf.device(FLAGS.device):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.device_percentage)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                              gpu_options=gpu_options)) as sess:
            cvae = build_cvae(sess, input_files,
                                 input_shape,
                                 FLAGS.latent_size,
                                 FLAGS.batch_size,
                                 epochs=FLAGS.epochs)
if __name__ == "__main__":
    main()
