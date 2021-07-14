import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

class VGGBlock(tf.keras.Model):
    def __init__(self, num_conv_layers=1, num_channels=64, *args, **kwargs) :
        super(VGGBlock, self).__init__(*args, **kwargs)
        self.conv_layers = self._generate_conv_layers(num_conv_layers, num_channels)
        self.pool_l = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")
        
    def _generate_conv_layers(self, num_conv_layers, num_channels):
        conv_layers = {}
        for _ in range(num_conv_layers):
            conv_layer = tf.keras.layers.Conv2D(filters=num_channels, kernel_size=3, strides=1, padding="same", activation="relu")
            conv_layers[conv_layer.name] = conv_layer
        return conv_layers
    
    def call(self, input):
        for conv_layer in self.conv_layers.values():
            input = conv_layer(input)
        return self.pool_l(input)
        
        
class VGG(tf.keras.Model):
    def __init__(self, conv_arch,*args, **kwargs):
        super(VGG, self).__init__(*args, **kwargs)
        self.block_sequence = self._generate_block_sequence(conv_arch)
        self.flatten_l = tf.keras.layers.Flatten()
        self.hidden_l1 = tf.keras.layers.Dense(units=4096, activation="relu")
        self.hidden_l2 = tf.keras.layers.Dense(units=4096, activation="relu")
        self.output_l = tf.keras.layers.Dense(units=10, activation="softmax")
        
    def _generate_block_sequence(self, conv_arch):
        block_sequence = {}
        for (num_conv_layers, num_channels) in conv_arch:
            block = VGGBlock(num_conv_layers, num_channels)      
            block_sequence[block.name] = block
        return block_sequence
           
    def call(self, input):
        for block in self.block_sequence.values():
            input = block(input)
        x = self.flatten_l(input)
        x = self.hidden_l1(x)
        x = self.hidden_l2(x) 
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
    print("Hola Mundo")
    print(train_images[0].shape)
    
    model = VGG( ((4,32),(8,64)) )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x=train_images, y=train_labels, validation_split=0.2, epochs=5)
    plot_history(history)
    results = model.evaluate(test_images, test_labels, verbose=2)
    print(results)


if __name__ == "__main__":
    main()