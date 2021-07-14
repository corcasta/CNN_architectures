import tensorflow as tf
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt

class LeNet5(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(LeNet5, self).__init__(*args, **kwargs)
        self.conv_l1 = tf.keras.layers.Conv2D(filters=6, kernel_size=5, strides=1, padding="same", activation="relu")
        self.pool_l1 =l1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")
        self.conv_l2 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, padding="valid", activation="relu")
        self.pool_l2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")
        self.flatten_l3 = tf.keras.layers.Flatten()
        self.hidden_l4 = tf.keras.layers.Dense(units=120, activation="relu")
        self.dropout_l4 = tf.keras.layers.Dropout(0.2)
        self.hidden_l5 = tf.keras.layers.Dense(units=84, activation="relu")
        self.dropout_l5 = tf.keras.layers.Dropout(0.2)
        self.output_l = tf.keras.layers.Dense(units=10, activation="softmax")

    def call(self, input):
        x = self.conv_l1(input)
        x = self.pool_l1(x)
        x = self.conv_l2(x)
        x = self.pool_l2(x)
        x = self.flatten_l3(x)
        x = self.hidden_l4(x)
        x = self.dropout_l4(x)
        x = self.hidden_l5(x)
        x = self.dropout_l5(x)
        return self.output_l(x)
    
    
def plot_history(model_history):
    history_df = pd.DataFrame(model_history.history)
    history_df["epoch"] = model_history.epoch
    history_df.plot.line(x="epoch", y=["accuracy", "val_accuracy"])
    plt.show()
        
def main():
    # Downloadfing MNIST dataset
    fashion_mnist = tf.keras.datasets.fashion_mnist
    
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    
    # Los labels representan numeros del 0-9
    class_names =['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] 
    
    # Normalicemos los features de entrenamientos y de evaluacion
    train_images = train_images/255.0
    test_images = test_images/255.0
    
    train_images = tf.expand_dims(train_images, axis=-1)
    test_images = tf.expand_dims(test_images, axis=-1)
    print("Hola Mundoc")
    print(train_images[0].shape)
    
    model = LeNet5()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x=train_images, y=train_labels, validation_split=0.2, epochs=25)
    plot_history(history)
    results = model.evaluate(test_images, test_labels, verbose=2)
    print(results)
    
    
if __name__ == "__main__":
    main()