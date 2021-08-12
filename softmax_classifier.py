import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras import callbacks


class Classifier(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(Classifier, self).__init__(*args, **kwargs)
        self.previous = tf.keras.layers.Flatten()
        self.hidden_l1 = tf.keras.layers.Dense(units=128, activation="relu")
        self.dropout_l1 = tf.keras.layers.Dropout(0.2   )
        self.hidden_l2 = tf.keras.layers.Dense(units=128, activation="relu")
        self.dropout_l2 = tf.keras.layers.Dropout(0.2)
        self.output_l = tf.keras.layers.Dense(units=10, activation="softmax")
    
    def call(self, inputs):
        x = self.previous(inputs)
        x = self.hidden_l1(x)
        x = self.dropout_l1(x)
        x = self.hidden_l2(x)
        x = self.dropout_l2(x)
        return self.output_l(x)
    
class CustomCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def set_model(self, model):
        self.model = model
        
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
    
    model = Classifier()
    #model.build((None,28*28))
    #print(model.summary())
    
    model.compile(optimizer="adam", 
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    model_filename = "training"
    checkpoint_path = os.path.join('saved_models/classifier', model_filename)
    #checkpoint_path = "/home/corcasta/Documents/AI/Test_1/saved_models/classifier"
    
    checkpoint_callback = CustomCheckpoint(filepath=checkpoint_path,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           verbose=1)
    history = model.fit(train_images, 
                        train_labels, 
                        validation_split=0.2, 
                        epochs=5, 
                        callbacks=[checkpoint_callback])
    
    plot_history(history)
    
    results = model.evaluate(test_images, test_labels, verbose=2)
    print(results)
    
    model.load_weights

if __name__ == "__main__":
    main()
