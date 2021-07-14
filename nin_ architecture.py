import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class NiNBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, kernel_size, strides, padding, *args, **kwargs):
        super(NiNBlock, self).__init__(*args, **kwargs)
        
        self.conv_l1 = tf.keras.layers.Conv2D(filters=num_channels, kernel_size=kernel_size, 
                                              strides=strides, padding=padding, activation="relu")
        
        self.conv_l2 = tf.keras.layers.Conv2D(filters=num_channels, kernel_size=1, strides=1, 
                                              padding="same", activation="relu")
        
        self.conv_l3 = tf.keras.layers.Conv2D(filters=num_channels, kernel_size=1, strides=1,
                                              padding="same", activation="relu")
   
    def call(self, input):
       x = self.conv_l1(input)
       x = self.conv_l2(x)
       return self.conv_l3(x)
       
        

class NiN(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(NiN, self).__init__(*args, **kwargs)
        
        self.nin_block_l1 = NiNBlock(num_channels=96, kernel_size=11, strides=4, padding="valid")
        self.max_pool_l1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")    
        
        self.nin_block_l2 = NiNBlock(num_channels=256, kernel_size=5, strides=1, padding="same")
        self.max_pool_l2 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")
        
        self.nin_block_l3 = NiNBlock(num_channels=384, kernel_size=3, strides=1, padding="same")
        self.max_pool_l3 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")
        
        self.nin_block_l4 = NiNBlock(num_channels=10, kernel_size=3, strides=1, padding="same")
        self.global_avg_pool_l4 = tf.keras.layers.GlobalAveragePooling2D()
        
    def call(self, input):
        x = self.nin_block_l1(input)
        x = self.max_pool_l1(x)
        x = self.nin_block_l2(x)
        x = self.max_pool_l2(x)
        x = self.nin_block_l3(x)
        x = self.max_pool_l3(x)
        x = self.nin_block_l4(x)
        return self.global_avg_pool_l4(x)

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
    print("Hola Mundo")
    print(train_images[0].shape)
    
    model = NiN()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x=train_images, y=train_labels, validation_split=0.2, epochs=5)
    plot_history(history)
    results = model.evaluate(test_images, test_labels, verbose=2)
    print(results)
    

if __name__ == "__main__":
    main()