import pickle

import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import text, sequence
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras.layers import Dense, Activation, Dropout
from keras import utils
from keras.models import load_model
import matplotlib.pyplot as plt
import Kratos
def split_data(filename):
    import string
    print("Spliting data")
    filename = filename
    news_df = pd.read_csv(filename, sep=",")
    news_df['CATEGORY'] = news_df.CATEGORY.map({'b': 1, 't': 2, 'e': 3, 'm': 4, 'p': 5, 's': 6})
    news_df['TITLE'] = news_df.TITLE.map(
        lambda x: x.lower().translate(str.maketrans('', '', string.punctuation))
    )
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        news_df['TITLE'],
        news_df['CATEGORY'],
        random_state=1
    )
    return X_train, X_test, y_train, y_test

def get_context(X_test):
    print("Tokenizing data")
    tokenize = pickle.load(open("tokenizer.pkl", "rb"))
    x_test = tokenize.texts_to_matrix(X_test)

    model = load_model('stored_model/my_model.h5')
    predictions = model.predict_classes(x_test)
    return predictions

def training_NN(X_train, X_test, y_train, y_test):
    print("Tokenizing data")
    max_words = 1000
    tokenize = text.Tokenizer(num_words=max_words, char_level=False)
    tokenize.fit_on_texts(X_train)
    pickle.dump(tokenize, open("tokenizer.pkl", "wb"))
    print("stored")
    x_train = tokenize.texts_to_matrix(X_train)
    x_test = tokenize.texts_to_matrix(X_test)
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)

    num_classes = np.max(y_train) + 1
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)
    # print('x_train shape:', x_train.shape)
    # print('x_test shape:', x_test.shape)
    # print('y_train shape:', y_train.shape)
    # print('y_test shape:', y_test.shape)
    batch_size = 32
    epochs = 2
    try:
        model = load_model('stored_model/my_model.h5')
        print("Model is loaded")
    except:
        print("model is training")
        from keras import Sequential
        model = Sequential()
        model.add(Dense(512, input_shape=(max_words,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_split=0.1)

        # model.save('stored_model/my_model.h5')

        plot_history(history)

    predictions = model.predict_classes(x_test)
    print("prediction is done")
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size, verbose=1)
    print('Test accuracy:', score[1])

    return predictions


def analyze_NN(filename):
    X_train, X_test, Y_train, Y_test = split_data(filename)
    predictions = training_NN(X_train, X_test, Y_train, Y_test)
    from Kratos import mapping
    predictions = mapping.map(predictions)
    print(predictions)


# def re_train_model(x_train, y_train, batch_size, epochs):
#     model = load_model('stored_model/my_model.h5')
#     model.fit(x_train, y_train,
#               batch_size=batch_size,
#               epochs=epochs,
#               verbose=1,
#               validation_split=0.1)
#     model.save('my_model.h5')


def plot_history(history):
    plt.style.use('ggplot')
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plot_history(history)