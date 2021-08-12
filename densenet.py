from logging import DEBUG
import tensorflow as tf
import numpy as np


class DenseBlock(tf.keras.Model):
    def __init__(self, num_convs, num_channels, *args, **kwargs):
        super(DenseBlock, self).__init__(*args, **kwargs)

        self.net = []
        for _ in range(num_convs):
            self.net.append(self.conv_block(num_channels))
            
    def call(self, input):
        x = input
        for dblock in self.net:
            for cblock in dblock:
                y = cblock(x)
                x = tf.concat([x,y], axis=-1)
        return x
        
    def conv_block(self, num_channels):
        block = [
            tf.keras.layers.BatchNormalization(),    
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=num_channels,
                                   kernel_size=3,
                                   padding="same")
        ]
        return block

 
class TransitionBlock(tf.keras.Model):
    def __init__ (self, num_channels, *args,**kwargs):
        super(TransitionBlock, self).__init__(*args, **kwargs)
        
        self.block = [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=num_channels, 
                                   kernel_size=1,
                                   strides=1),
            tf.keras.layers.AvgPool2D(pool_size=2, strides=2)
        ]
        
    def call(self, input):
        x = input
        for layer in self.block:
            x = layer(x)
        return x
        
        
class DenseNet(tf.keras.Model):
    def __init__(self, num_channels, growth_rate, num_convs_in_dense_blocks, *args, **kwargs):
        super(DenseNet, self).__init__(*args, **kwargs)
        self.l1 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=1, padding="same")
        self.l2 = tf.keras.layers.BatchNormalization()
        self.l3 = tf.keras.layers.ReLU()
        self.l4 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")
        self.block_sequence = self.generate_block_sequence(num_channels,
                                                           growth_rate, 
                                                           num_convs_in_dense_blocks)
        self.l5 = tf.keras.layers.BatchNormalization()
        self.l6 = tf.keras.layers.ReLU()
        self.l7 = tf.keras.layers.GlobalAveragePooling2D()
        self.l8 = tf.keras.layers.Dense(units=10, activation="softmax")
        
    def generate_block_sequence(self, num_channels, growth_rate, num_convs_in_dense_blocks):
        block_sequence = []
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            block_sequence.append(DenseBlock(num_convs, growth_rate))
            num_channels += num_convs * growth_rate
            if i != len(num_convs_in_dense_blocks) - 1:
                num_channels //= 2
                block_sequence.append(TransitionBlock(num_channels)) 
        return block_sequence    
        
                
    def call(self, input):
        x = self.l1(input)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        for block in self.block_sequence:
            x = block(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        return x
        
        
    

def main():

    """
    #block = DenseBlock(2, 10)
    #x = np.random.uniform(size=(1,28,28,1))
    #y = block(x)
    #block = TransitionBlock(10)
    #block(y)
    #block.summary()
    
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4,4,4,4]
    model = DenseNet(num_channels, growth_rate, num_convs_in_dense_blocks)
    model.build((None, 28, 28, 1))
    model.summary()
    """
    
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
    
    #We can tune these hyperparameter for faster or more precise model training
    num_channels, growth_rate = 32, 16
    num_convs_in_dense_blocks = [2,2,2,2]
    #**************************************************************************
    model = DenseNet(num_channels, growth_rate, num_convs_in_dense_blocks)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x=train_images, y=train_labels, validation_split=0.2, epochs=5)
    plot_history(history)
    results = model.evaluate(test_images, test_labels, verbose=2)
    print(results)
        
if __name__ == "__main__":
    main()