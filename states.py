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
from utils import evaluate_model, get_dataset, create_callbacks

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
          optimizer = tfa.optimizers.LAMB(learning_rate=learning_rate)
          model.optimizer = optimizer
          model.compile(loss=MyCustomLoss(), metrics=tf.keras.metrics.CategoricalAccuracy())
          return model

    def run(self):
        self.log("Transition to compute")
        CONFIG_FILE = "config.yml"
        file_path = f"/mnt/input/{CONFIG_FILE}"
        config_f = bios.read(file_path)
        self.store('config_f', config_f)
        X = pd.read_csv(f"/mnt/input/{config_f['federated_sti']['dataset']['train']}", index_col=0)
        Y_train = pd.read_csv(f"/mnt/input/{config_f['federated_sti']['dataset']['labels']}", index_col=0).squeeze("columns")
        feature_size = X.shape[1]
        self.store('chunk_size', X.shape[1])
        _x = X.to_numpy(dtype=np.int32)
        _y = Y_train.to_numpy()
        BATCH_SIZE = 2
        x_train, x_valid, y_train, y_valid = train_test_split(_x, _y, test_size=0.10, random_state=1, shuffle=True)
        steps_per_epoch = 2*x_train.shape[0]//BATCH_SIZE
        validation_steps = 2*x_valid.shape[0]//BATCH_SIZE
        train_dataset = get_dataset(x_train, 0, feature_size, 0, 0, BATCH_SIZE)
        valid_dataset = get_dataset(x_valid, 0, feature_size, 0, 0, BATCH_SIZE, training=False)
        self.store('steps_per_epoch', steps_per_epoch)
        self.store('validation_steps', validation_steps)
        self.store('train_dataset', train_dataset)
        self.store('valid_dataset', valid_dataset)
        self.store('iteration', 0)
        model = self.create_model()
        self.store('model', model)
        self.store('NUM_EPOCHS', 10)
        callbacks = create_callbacks()
        self.store('callbacks', callbacks)
        return 'compute'

@app_state('compute')
class ComputeState(AppState):

    def register(self):
        self.register_transition('obtain_weights', role=Role.PARTICIPANT)
        self.register_transition('aggregate', role=Role.COORDINATOR)

    def run(self):
        self.log("Start computation")
        inChannel = 3
        weight_decay = 0.00001
        embed_dim = 64
        ff_dim = 32
        regularization_coef_l1 = 1e-4
        dropout_rate = 0.25
        attention_range = 0
        missing_perc = 0.1
        callbacks = self.load('callbacks')
        steps_per_epoch = self.load('steps_per_epoch')
        validation_steps = self.load('validation_steps')
        train_dataset = self.load('train_dataset')
        valid_dataset = self.load('valid_dataset')
        model = self.load('model')
        iteration = self.load('iteration')
        if iteration >= 1 and not self.is_coordinator:
            weights = self.load('weights')
            model.layers[0].set_weights(weights[0])
            model.layers[-1].set_weights(weights[1])
        history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=1,
            validation_data=valid_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks, verbose=1)
        self.send_data_to_coordinator(model, send_to_self=True)
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
    updated_weights = self.await_data(n = 1)
    self.store('weights', updated_weights)
    return "compute"


@app_state('aggregate')
class AggregateState(AppState):

    def register(self):
        self.register_transition('compute', Role.COORDINATOR)
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
    def evaluate(self, model, mode, test_dataset, test_Y, bin_labels):
        evaluate_model(self, model, mode, test_dataset, test_Y, bin_labels)
    def run(self):
        self.log("Start aggregation")
        data = self.gather_data()
        weights1 = self.aggregate_weights(data, 0)
        weights2 = self.aggregate_weights(data, -1)
        if self.is_coordinator:
            model = self.load('model')
            model.layers[0].set_weights(weights1)
            model.layers[-1].set_weights(weights2)
            self.store('model', model)
        iteration = self.load("iteration")
        NUM_EPOCHS = self.load('NUM_EPOCHS')
        if iteration >= NUM_EPOCHS:
            config_f = self.load('config_f')
            test_X = pd.read_csv(f"/mnt/input/{config_f['federated_sti']['dataset']['test']}", index_col=0)
            test_Y = pd.read_csv(f"/mnt/input/{config_f['federated_sti']['dataset']['test_labels']}", index_col=0).squeeze("columns")
            bin_labels = pd.read_csv(f"/mnt/input/{config_f['federated_sti']['dataset']['bin_labels']}")
            bin_labels = bin_labels.iloc[0, 0]
            test_dataset = (test_X.to_numpy(dtype=np.int32), test_Y.to_numpy())
            self.evaluate(model, "aggregated", test_dataset, test_Y, bin_labels)
            model.save_weights('mnt/output/trained_model_weights')
            return 'terminal'
        else:
            self.broadcast_data([weights1, weights2], send_to_self = False)
            return 'compute'