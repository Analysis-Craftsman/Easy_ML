from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np

def deep_model_maker(train_x, train_y, test_x, test_y, cnt) :
    model = Sequential()

    model.add(Dense(30, input_shape=(cnt,), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(6, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='softmax'))

    # categorical_crossentropy
    # mean_squared_error
    model.compile(loss='mean_squared_error', optimizer="rmsprop", metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=300, epochs=100)
    
    score = model.evaluate(test_x, test_y)
    print('loss =', int(score[0]*100))
    print('accuracy =', int(score[1]*100))


def deep_learing(path) :
    excel = pd.read_excel( path, "DATASET" )
    df = pd.DataFrame( excel )

    data_slice_bid = int( len(df) * 0.7 )

    train_x, train_y = df.iloc[:data_slice_bid][df.columns[:-1]], df.iloc[:data_slice_bid][df.columns[-1]]
    test_x, test_y = df.iloc[data_slice_bid:-1][df.columns[:-1]], df.iloc[data_slice_bid:-1][df.columns[-1]]

    deep_model_maker( np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y), int(len(df.columns)-1) )


if __name__ == "__main__":

    path = "./you_path/030.xlsx"
    deep_learing(path)
