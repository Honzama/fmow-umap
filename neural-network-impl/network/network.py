
import sys
import os.path
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
#from keras.applications import EfficientNetB0
import efficientnet.keras as efn
from data.fmow import Fmow
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

from params import fmow_category_names, fmow_image_size, save_model_file

# load_fmow function load normalized train_rgb and val_rgb data

def load_fmow():
    train = Fmow("train_rgb")
    test = Fmow("val_rgb")

    data_train, labels_train = train.get_normalized()
    data_test, labels_test = test.get_normalized()

    data, labels = test.get()

    return data_train, labels_train, data_test, labels_test, labels

# accuracy_graph function shows graph of train and validation accuracy of given model history

def accuracy_graph(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

# get_model_effnetB0 function creates ands returns EfficientNetB0 network model

def get_model_effnetB0(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    outputs = efn.EfficientNetB0(include_top=True, weights=None, classes=num_classes)(inputs)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

# load_model_effnet function loads and returns saved model if it exist, or None if it not exist

def load_model_effnet():
    if os.path.isfile(str(save_model_file)):
        return keras.models.load_model(str(save_model_file))
    else:
        return None

# train_model function train and save given model

def train_model(model, file_id, x_train, y_train, x_test, y_test, epochs):
    model.summary()

    hist = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=1)

    model.save(str(save_model_file))

    accuracy_graph(hist)

# test_model function runs trained model on test data and print results

def test_model(model, x_test, y_test):
    model.summary()

    score = model.evaluate(x_test, y_test, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

# measure_model function creates confusion matrix and classification report from model

def measure_model(model, x_test, classes):
    model.summary()

    predictions = model.predict(x_test, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    conf_matrix = confusion_matrix(y_true=classes, y_pred=predicted_classes)
    conf_matrix_graph(conf_matrix)

    report = classification_report(classes, predicted_classes, target_names=fmow_category_names)
    print("Classification Report")
    print(report)

# conf_matrix_graph function shows graph of confusion matrix

def conf_matrix_graph(conf_matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_matrix)
    plt.title("Confusion matrix of the EfficientNetB0")
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":

    # fmow variables
    x_train, y_train, x_test, y_test, classes = load_fmow()
    num_classes = len(fmow_category_names)

    # net variables
    batch_size = 32
    epochs = 10
    input_shape = (fmow_image_size, fmow_image_size, 3)

    # model, file_id = get_model_effnetB0(input_shape, num_classes)
    # train_model(model, x_train, y_train, x_test, y_test, epochs)

    model = load_model_effnet()
    #test_model(model, x_test, y_test)

    measure_model(model, x_test, classes)

    #sys.exit()
