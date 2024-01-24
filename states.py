from FeatureCloud.app.engine.app import AppState, app_state, Role, LogLevel, SMPCOperation
import time
import bios
import os
import tensorflow as tf
import pandas as pd
import math
import numpy as np
import tensorflow.keras.backend as K
from models.model import SplitTransformer
from models.loss import MyCustomLoss
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error
from sklearn.linear_model import LassoCV, ElasticNetCV
from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa
from tensorflow import keras
from tqdm import tqdm
from utils import evaluate_model

@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('compute', Role.BOTH)

    def create_model(self):
          chunk_size = self.load('chunk_size')
          learning_rate = 0.01
          model =  SplitTransformer(embed_dim=256,
              num_heads=40,
              attn_block_repeats=1,
              chunk_size=chunk_size,
              activation="gelu",
              attention_range=0)
          #optimizer = tfa.optimizers.LAMB(learning_rate=learning_rate)
          optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
          model.compile(optimizer=optimizer, loss=MyCustomLoss(), metrics=tf.keras.metrics.CategoricalAccuracy())
          return model

    def run(self):
        self.log("Transition to compute")
        CONFIG_FILE = "config.yml"
        file_path = f"/mnt/input/{CONFIG_FILE}"
        config_f = bios.read(file_path)
        self.store('config_f', config_f)
        X = pd.read_csv(f"/mnt/input/{config_f['federated_sti']['dataset']['train']}", index_col=0)
        Y_train = pd.read_csv(f"/mnt/input/{config_f['federated_sti']['dataset']['labels']}", index_col=0).squeeze("columns")
        self.store('Y_train', Y_train)
        self.store('feature_size', X.shape[1])
        self.store('chunk_size', X.shape[1])
        _x = X.to_numpy(dtype=np.int32)
        _y = Y_train.to_numpy()
        self.store('_x', _x)
        self.store('_y', _y)
        self.store('iteration', 0)
        model = self.create_model()
        self.store('model', model)
       # physical_devices = tf.config.list_physical_devices('GPU')
       # if len(physical_devices) == 0:
       #     raise RuntimeError("No GPU devices found.")
       # else:
       #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
        return 'compute'

@app_state('compute')
class ComputeState(AppState):

    def register(self):
        self.register_transition('obtain_weights', role=Role.PARTICIPANT)
        self.register_transition('aggregate', role=Role.COORDINATOR)

    @tf.function()
    def add_attention_mask(self, X_sample, y_sample):
      depth = 3
      mask_size = tf.cast(X_sample.shape[0]*0.5, dtype=tf.int32)
      mask_idx = tf.reshape(tf.random.shuffle(tf.range(X_sample.shape[0]))[:mask_size], (-1, 1))
      updates = tf.math.add(tf.ones(shape=(mask_idx.shape[0]), dtype=tf.int32), 1)
      X_masked = tf.tensor_scatter_nd_update(X_sample, mask_idx, updates)
      return tf.one_hot(X_masked, depth), tf.one_hot(y_sample, depth-1)

    def map_values_1(self, x):
        return 0 if (x == 0 or x == 1) else 1

    def map_values_2(self, x):
        return 0 if (x == 0 or x == 2) else 1

    def get_dataset(self, x, chunk_start, chunk_end, start_offset, end_offset, batch_size, training=True):
      AUTO = tf.data.AUTOTUNE

      _x = np.empty((x.shape[0] * 2, chunk_end-chunk_start), dtype=x.dtype)

      map_values_1_vec = np.vectorize(self.map_values_1)
      map_values_2_vec = np.vectorize(self.map_values_2)

      _x[0::2] = map_values_1_vec(x[:, chunk_start:chunk_end])
      _x[1::2] = map_values_2_vec(x[:, chunk_start:chunk_end])
      new_chunk_end = _x.shape[1]

      dataset = tf.data.Dataset.from_tensor_slices((_x,
                                                    _x[:, start_offset:new_chunk_end-end_offset]))

      if training:
        dataset = dataset.shuffle(_x.shape[0], reshuffle_each_iteration=True)
        dataset = dataset.repeat()

      # Add Attention Mask
      dataset = dataset.map(self.add_attention_mask, num_parallel_calls=AUTO, deterministic=False)

      # Prefetech to not map the whole dataset
      dataset = dataset.prefetch(AUTO)

      dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=AUTO)

      return dataset

    def create_callbacks(self, metric = "val_loss"):
        reducelr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor= metric,
            mode='auto',
            factor=0.2,
            patience=3,
            verbose=0
        )

        earlystop = tf.keras.callbacks.EarlyStopping(
            monitor= metric,
            mode='auto',
            patience= 20,
            verbose=1,
            restore_best_weights=True
        )

        callbacks = [
                     reducelr,
                     earlystop]

        return callbacks
    def evaluate(self, model, mode, test_dataset, test_Y):
        evaluate_model(self, model, mode, test_dataset, test_Y)

    def run(self):
        self.log("Start computation")
        feature_size = self.load('feature_size')
        inChannel = 3
        weight_decay = 0.00001
        embed_dim = 64
        ff_dim = 32
        regularization_coef_l1 = 1e-4
        dropout_rate = 0.25
        attention_range = 0
        chunk_size = self.load('chunk_size')
        missing_perc = 0.1
        NUM_EPOCHS = 50
        BATCH_SIZE = 2
        _x = self.load('_x')
        _y = self.load('_y')
        x_train, x_valid, y_train, y_valid = train_test_split(_x, _y, test_size=0.10,
                                            random_state=1,
                                            shuffle=True)
        steps_per_epoch = 2*x_train.shape[0]//BATCH_SIZE
        validation_steps = 2*x_valid.shape[0]//BATCH_SIZE
        train_dataset = self.get_dataset(x_train, 0, feature_size, 0, 0, BATCH_SIZE)
        valid_dataset = self.get_dataset(x_valid, 0, feature_size, 0, 0, BATCH_SIZE, training=False)
        K.clear_session()
        callbacks = self.create_callbacks()
        model = self.load('model')
        history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=NUM_EPOCHS,
            validation_data=valid_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks, verbose=1, batch_size=5)
        test_dataset = self.load('test_dataset')
        test_Y = self.load('test_Y')
        self.send_data_to_coordinator(model, send_to_self=True)
        iteration = self.load('iteration')
        iteration += 1
        self.store("iteration", iteration)
        if self.is_coordinator:
            return "aggregate"
        else:
            return "obtain_weights"

@app_state("obtain_weights")
class ObtainWeights(AppState):
  def register(self):
    self.register_transition("compute", Role.BOTH)

  def run(self):
    updated_model = self.await_data(n = 1)
    self.store('model', updated_model)
    return "compute"


@app_state('aggregate')
class AggregateState(AppState):

    def register(self):
        self.register_transition('obtain_weights', Role.COORDINATOR)
        self.register_transition("terminal", Role.COORDINATOR)

    def map_values_1(self, x):
        return 0 if (x == 0 or x == 1) else 1

    def map_values_2(self, x):
        return 0 if (x == 0 or x == 2) else 1

    def aggregate_weights(self, model_list, layer_index):
        parameters = []
        biases = []
        all_weights = []
        for i in range(len(model_list)):
            model_weights = model_list[i].layers[layer_index].get_weights()
            model_weights = [np.array(w) for w in model_weights]

            if i == 0:
                all_weights = [[] for _ in range(len(model_weights))]
            for j in range(len(model_weights)):
                all_weights[j].append(model_weights[j])
        updated_weights = [np.mean(weights, axis=0) for weights in all_weights]
        return updated_weights
    def evaluate(self, model, mode, test_dataset, test_Y):
        evaluate_model(self, model, mode, test_dataset, test_Y)
    def run(self):
        self.log("Start aggregation")
        data = self.gather_data()
        config_f = self.load('config_f')
        test_X = pd.read_csv(f"/mnt/input/{config_f['federated_sti']['dataset']['test']}", index_col=0)
        test_Y = pd.read_csv(f"/mnt/input/{config_f['federated_sti']['dataset']['test_labels']}", index_col=0).squeeze("columns")
        test_dataset = (test_X.to_numpy(dtype=np.int32), test_Y.to_numpy())
        #for i, model in enumerate(data):
            #self.evaluate(model, "local" + str(i), test_dataset, test_Y)
        model = data[0]
        self.evaluate(model, "local", test_dataset, test_Y)
        model.layers[0].set_weights(self.aggregate_weights(data, 0))
        model.layers[-1].set_weights(self.aggregate_weights(data, -1))
        self.evaluate(model, "aggregated", test_dataset, test_Y)
        iteration = self.load("iteration")
        if iteration >= 1:
            model.save_weights('mnt/output/trained_model_weights')
            return 'terminal'
        else:
            self.broadcast_data(model, send_to_self = True)
            return 'obtain_weights'