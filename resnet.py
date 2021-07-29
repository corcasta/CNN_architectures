import tensorflow as tf
import numpy as np

class ResidualBlock(tf.keras.Model):
    def __init__(self, num_channels, conv_layer_1x1=False, strides=1, *args, **kwargs):
        super(ResidualBlock, self).__init__(*args, **kwargs)   
        self.conv_l1 = tf.keras.layers.Conv2D(filters=num_channels, 
                                              kernel_size=3, 
                                              padding="same", 
                                              strides=strides)
        self.batch_norm_l2 = tf.keras.layers.BatchNormalization()
        self.conv_l3 = tf.keras.layers.Conv2D(filters=num_channels, 
                                              kernel_size=3, 
                                              padding="same", 
                                              strides=1)
        self.batch_norm_l4 = tf.keras.layers.BatchNormalization()
        if conv_layer_1x1:
            self.conv_l0 = tf.keras.layers.Conv2D(filters=num_channels, 
                                                  kernel_size=1, 
                                                  padding="same", 
                                                  strides=strides)
        else:
            self.conv_l0 = None
        self.relu_l = tf.keras.layers.ReLU()
         
    def call(self, input):
        x = self.conv_l1(input)
        x = self.batch_norm_l2(x)
        x = self.relu_l(x)
        x = self.conv_l3(x)
        x = self.batch_norm_l4(x)
        if self.conv_l0:
            input = self.conv_l0(input) 
        x = tf.add(x, input)
        return self.relu_l(x)
        

class ResNet(tf.keras.Model):
    def __init__(self, conv_arch, num_channels=64, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        self.conv_l1 = tf.keras.layers.Conv2D(filters=num_channels, 
                                              kernel_size=7, 
                                              padding="same", 
                                              strides=1)
        self.batchnorm_l2 = tf.keras.layers.BatchNormalization()
        self.maxpool_l3 = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="valid")
        self.block_sequence = self._generate_block_sequence(conv_arch=conv_arch)
        self.global_avr_l = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten_l = tf.keras.layers.Flatten()
        self.output_l = tf.keras.layers.Dense(units=10, activation="softmax")

    def _resnet_block(self, num_channels, num_residuals, first_block=False):
        resnet_block = {}
        for i in range(num_residuals):
            if i == 0 and not first_block:
                residual_block = ResidualBlock(num_channels, conv_layer_1x1=True, strides=2)
                resnet_block[residual_block.name] = residual_block
            else:
                residual_block = ResidualBlock(num_channels)
                resnet_block[residual_block.name] = residual_block
        return resnet_block
    
    def _generate_block_sequence(self, conv_arch):
        counter = 0
        block_squence = {}
        for tuple_ in conv_arch:
            if len(tuple_) == 3:
                resnet_block = self._resnet_block(num_channels=tuple_[0], 
                                                  num_residuals=tuple_[1], 
                                                  first_block=tuple_[2])
            else:
                resnet_block = self._resnet_block(num_channels=tuple_[0],
                                                  num_residuals=tuple_[1])
            block_squence["resnet_block_{}".format(counter)] = resnet_block
            counter += 1
        return block_squence
                
    def call(self, input):
        x = self.conv_l1(input)
        x = self.batchnorm_l2(x)
        x = self.maxpool_l3(x) 
        for resnet_block in self.block_sequence.values():
            for residual_block in resnet_block.values():
                x = residual_block(x) 
        x = self.global_avr_l(x)
        x = self.flatten_l(x)
        return self.output_l(x)

def plot_history(model_history):
    history_df = pd.DataFrame(model_history.history)
    history_df["epoch"] = model_history.epoch
    history_df.plot.line(x="epoch", y=["accuracy", "val_accuracy"])
    plt.show()
    
def main():
    """
    X = np.random.uniform(size=(4,6,6,3))
    block = ResidualBlock(num_channels=6, conv_layer_1x1=True, strides=2)
    block.build((None, 6, 6, 3))
    block.summary()
    print(block(X).shape)
    """
    
    """
    X = np.random.uniform(size=(1,224,224,1))
    resnet = ResNet()
    resnet.build((None, 224, 224, 1))
    resnet.summary()
    print(resnet(X).shape)
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
    
    model = ResNet(conv_arch=[(64, 2, True), (128, 2), (256, 2), (512, 2)])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x=train_images, y=train_labels, validation_split=0.2, epochs=5)
    plot_history(history)
    results = model.evaluate(test_images, test_labels, verbose=2)
    print(results) 
    
    
    
    
if __name__ == "__main__":
    main()