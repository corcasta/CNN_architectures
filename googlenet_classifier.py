import tensorflow as tf

class InceptionBlock(tf.keras.Model):
    def __init__(self, c1, c2, c3, c4, *args, **kwargs):
        super(InceptionBlock, self).__init__(*args, **kwargs)
        # First path
        self.path1_l1 = tf.keras.layers.Conv2D(filters=c1, kernel_size=1, strides=1, padding="valid", activation="relu", name="path1_l1")
        
        # Second path
        self.path2_l1 = tf.keras.layers.Conv2D(filters=c2, kernel_size=1, strides=1, padding="valid", activation="relu", name="path2_l1")
        self.path2_l2 = tf.keras.layers.Conv2D(filters=c2, kernel_size=3, strides=1, padding="same", activation="relu", name="path2_l2")
        
        # Third path
        self.path3_l1 = tf.keras.layers.Conv2D(filters=c3, kernel_size=1, strides=1, padding="valid", activation="relu", name="path3_l1")
        self.path3_l2 = tf.keras.layers.Conv2D(filters=c3, kernel_size=5, strides=1, padding="same", activation="relu", name="path3_l2")    
    
        # Fourth
        self.path4_l1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same", name="path4_l1")
        self.path4_l2 = tf.keras.layers.Conv2D(filters=c4, kernel_size=1, strides=1, padding="valid", activation="relu", name="path4_l2") 

    def call(self, input):
        # First path
        x1 = self.path1_l1(input)
        # Second path
        x2 = self.path2_l2(self.path2_l1(input))
        # Third path
        x3 = self.path3_l2(self.path3_l1(input))
        # Fourth path 
        x4 = self.path4_l2(self.path4_l1(input))
        # Concatenate the outputs
        return tf.concat(values=[x1, x2, x3, x4], axis=-1)
        

class GoogleNet(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(GoogleNet, self).__init__(*args, **kwargs)
        
        
def main():
    block = InceptionBlock(1,1,1,1)
    block.build(input_shape=(None, 28, 28, 1))
    block.summary()
    
if __name__ == "__main__":
    main()
        
        