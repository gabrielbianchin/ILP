import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import metrics
import math
import sys
import gc
import pickle

gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

PATH_MODEL_BASE = "../../Models/resnet50-random-{}-run-{}.keras"
PATH_PREDICTIONS_BASE = "../../Predictions/resnet50-random-{}-run-{}-{}.csv"
PATH_PLOT = "../../plots/resnet50-random-{}-run-{}.pkl"
PATH_IMGS = "../../UTKFace"
PATH_TRAIN = "../../Base/train.csv"
PATH_VALID = "../../Base/val.csv"
PATH_TEST = "../../Base/test.csv"

RANDOM_INSTANCE = "../../RandomTrainingSet/random-{}-run-{}.txt"

def compute_metrics(y_true, y_pred, names, path_csv):
  print('MAE:', metrics.mean_absolute_error(y_true = y_true, y_pred = y_pred))

  sum = 0
  for age in np.unique(y_true):
    lines = np.where(y_true == age)[0]
    sum += metrics.mean_absolute_error(y_true = y_true[lines], y_pred = y_pred[lines])

  print('MAE balanceado:', sum/len(np.unique(y_true)))

  pd.DataFrame({'image': names, 'predictions': np.reshape(y_pred, (-1))}).to_csv(path_csv, index=False)

def process_dataframe(df):
  gender, race = [], []
  for i in df.filename.values:
    meta_data = i.split('_')
    try:
      race.append(int(meta_data[2]))
      gender.append(int(meta_data[1]))
    except:
      race.append(-1)
      gender.append(-1)
      print(meta_data)
  df['gender'] = gender
  df['race'] = race

df_val = pd.read_csv(PATH_VALID)
df_test = pd.read_csv(PATH_TEST)
process_dataframe(df_val)
process_dataframe(df_test)

df_val["filename"][df_val["filename"] == "39_0_1_20170116174525125.jpg.chip.jpg"] = "39_1_20170116174525125.jpg.chip.jpg"

generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input)

val_generator = generator.flow_from_dataframe(
        df_val, directory=PATH_IMGS, x_col='filename', y_col=['age'], image_size=(128, 128),
        target_size=(128, 128), class_mode='raw', batch_size=16, shuffle=False)

test_generator = generator.flow_from_dataframe(
        df_test, directory=PATH_IMGS, x_col='filename', y_col=['age'], image_size=(128, 128),
        target_size=(128, 128), class_mode='raw', batch_size=16, shuffle=False)

SIZE = int(sys.argv[1])
N_RUN = int(sys.argv[2])
print('SIZE:', SIZE, ' N_RUN:', N_RUN)
PATH_MODEL = PATH_MODEL_BASE.format(SIZE, N_RUN)
df_train = pd.read_csv(PATH_TRAIN)
process_dataframe(df_train)
df_train = pd.read_csv(RANDOM_INSTANCE.format(SIZE, N_RUN))
df_train["filename"][df_train["filename"] == "61_1_1_20170109142408075.jpg.chip.jpg"] = "61_1_20170109142408075.jpg.chip.jpg"
df_train["filename"][df_train["filename"] == "61_1_3_20170109150557335.jpg.chip.jpg"] = "61_1_20170109150557335.jpg.chip.jpg"
train_generator = generator.flow_from_dataframe(
          df_train, directory=PATH_IMGS, x_col='filename', y_col=['age'], image_size=(128, 128),
          target_size=(128, 128), class_mode='raw', batch_size=16, shuffle=True)

backbone = tf.keras.applications.ResNet50(include_top=False, input_shape=(128,128,3), pooling='avg')

for layer in backbone.layers:
  layer.trainable = True

model = tf.keras.Sequential([
      backbone,
      tf.keras.layers.Dense(1, activation=None)
])

model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001), loss=['mae'])
mc = tf.keras.callbacks.ModelCheckpoint(PATH_MODEL, save_best_only=True, verbose=1)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1, patience=20, min_delta=0.01)
lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=10)

history = model.fit(train_generator, epochs=200, validation_data=val_generator, callbacks=[es, mc, lr], verbose=1, batch_size=16)

SAVE_PATH = PATH_PREDICTIONS_BASE.format(SIZE, N_RUN, "val")
pred = model.predict(val_generator, verbose=1)
compute_metrics(y_true = val_generator.labels, y_pred = pred, names = val_generator.filenames, path_csv = SAVE_PATH)

SAVE_PATH = PATH_PREDICTIONS_BASE.format(SIZE, N_RUN, "test")
pred = model.predict(test_generator, verbose=1)
compute_metrics(y_true = test_generator.labels, y_pred = pred, names = test_generator.filenames, path_csv = SAVE_PATH)

with open(PATH_PLOT.format(SIZE, N_RUN), 'wb') as f:
  pickle.dump(history.history, f)

del model
gc.collect()

