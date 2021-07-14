import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.backend import dtype

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class LinearRegressionModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_l1 = tf.keras.layers.Dense(units=64, activation="relu")
        self.hidden_l2 = tf.keras.layers.Dense(units=64, activation="relu")
        self.output_l = tf.keras.layers.Dense(units=1)
        
    def call(self, input, **kwargs):
        x = self.hidden_l1(input)
        x = self.hidden_l2(x)
        return self.output_l(x)
    
def norm(dataset, train_dataset):
    train_dataset_stats = train_dataset.describe().transpose()
    return (dataset-train_dataset_stats["mean"])/train_dataset_stats["std"]
 
def plot_history(model_history):
    history = pd.DataFrame(model_history.history)
    history["epoch"] = model_history.epoch

    history.plot.line(x="epoch", y=["mae", "val_mae"])
    plt.show()
    
    
def main():
    # Download dataset if does not exist in following path and return the path itself: "/home/USER/.keras/datasets/auto-mpg.data"
    dataset_path = tf.keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")     
     
    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                    'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                        na_values = "?", comment='\t',
                        sep=" ", skipinitialspace=True)
    
    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    
    # Origin column represent categorical values, it need to be change as one-hot
    origin_col = dataset.pop(item="Origin")
    dataset["USA"] = (origin_col == 1)*1.0
    dataset["Europe"] = (origin_col == 1)*1.0
    dataset["Japan"] = (origin_col == 1)*1.0
    
    # Feature values 
    train_dataset = dataset.sample(frac=0.8)
    test_dataset = dataset.drop(index=train_dataset.index)
    
    # Value to be predicted
    train_labels = train_dataset.pop("MPG")
    test_labels = test_dataset.pop("MPG")
    
    normed_train_dataset = norm(train_dataset, train_dataset)
    normed_test_dataset = norm(test_dataset, train_dataset)
    #print(train_dataset.head())
    #print(normed_train_dataset.head())
    
    #print(len(train_dataset.keys()))
    model = LinearRegressionModel()
    
    # Se construye el modelo unicamente para poder acceder al resumen de la estructura
    model.build(input_shape=(None,9))
    print("*********** DEBUG ************")
    print(model.non_trainable_weights)
    print("*********** DEBUG ************")
    # model.summary()
    model.compile(loss="mse", 
                  optimizer="Adam", 
                  metrics=['mae', 'mse'])
    
    history = model.fit(normed_train_dataset, train_labels, epochs=1000, validation_split=0.2, verbose=0)
    plot_history(history)
    
    print("Lenght of  test_dataset: ", len(normed_test_dataset.index))
    results = model.evaluate(x=normed_test_dataset, y=test_labels, return_dict=True, verbose=1)
    print(results)
     
if __name__ == "__main__":
    main()
    
    