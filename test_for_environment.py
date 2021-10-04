import tensorflow as tf
if __name__ == "__main__":
    print("Is current tensorflow avaiable?{}".format(tf.config.list_physical_devices('GPU')))
    print("The tensorflow version is {}".format(tf.test.is_gpu_available()))