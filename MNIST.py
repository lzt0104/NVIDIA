from tensorflow.keras.datasets import mnist
# the data, split between train and validation sets
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
x_train.shape
x_valid.shape
