from FeatureCloud.app.engine.app import AppState, app_state, Role, LogLevel, SMPCOperation
import time
import bios
import os
import tensorflow as tf
import pandas as pd
import math
import numpy as np
import allel
import tensorflow.keras.backend as K
from models.model import SplitTransformer
from models.loss import MyCustomLoss
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.model_selection import KFold,StratifiedKFold
from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa
from tensorflow import keras
from scipy.spatial.distance import squareform

from tqdm import tqdm

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
        new_data_header = ""
        genotypes = pd.read_csv(f"/mnt/input/{config_f['federated_sti']['dataset']['train']}", comment='#', sep='\t', header = 1, index_col='Sample_id', dtype={'Sample_id':str})
        headers = genotypes.columns[:]
        self.save('headers', headers)
        pedigree = pd.read_csv(f"/mnt/input/{config_f['federated_sti']['dataset']['labels']}",  sep='\t', index_col='Individual ID')
        Y_train = pedigree.loc[genotypes.index]['Population']
        self.store('Y_train', Y_train)
        X = genotypes[genotypes.index.isin(Y_train.index)]
        X = X.replace({
            '0|0': 0,
            '0|1': 1,
            '1|0': 2,
            '1|1': 3
        })
        self.store('feature_size', X.shape[1])
        self.store('chunk_size', X.shape[1])
        r = allel.rogers_huff_r(X.T)
        LD = squareform(r ** 2)
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
        LD_max_freqs = np.amax(LD, axis=1)
        bin_labels = np.digitize(LD_max_freqs, bins=bins, right=True)
        self.store('bin_labels', bin_labels)
        fold = 0
        self.store('fold', fold)
        _x = X[X.index.isin(Y_train.index)].to_numpy()
        _y = Y_train.to_numpy()
        self.store('_x', _x)
        self.store('_y', _y)
        N_SPLITS=2
        self.store('N_SPLITS', N_SPLITS)
        self.store('iteration', 0)
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=2022)
        self.store('kf', kf)
        model = self.create_model()
        self.store('model', model)
        self.store('weights', None)
        self.save('accuracies', [])
        return 'compute'

@app_state('compute')
class ComputeState(AppState):

    def register(self):
        self.register_transition('obtain_weights', role=Role.PARTICIPANT)
        self.register_transition('aggregate', role=Role.COORDINATOR)

    @tf.function()
    def add_attention_mask(self, X_sample, y_sample):
      depth = 3
      mask_size = tf.cast(X_sample.shape[0]*0.5, dtype=tf.int64)
      mask_idx = tf.reshape(tf.random.shuffle(tf.range(X_sample.shape[0]))[:mask_size], (-1, 1))
      updates = tf.math.add(tf.ones(shape=(mask_idx.shape[0]), dtype=tf.int64), 1)
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

    def create_callbacks(self, kfold=0, metric = "val_loss"):
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
        NUM_EPOCHS = 1
        kf = self.load('kf')
        BATCH_SIZE = 5
        _x = self.load('_x')
        _y = self.load('_y')
        train_index, test_index = next(kf.split(_x))
        fold = self.load('fold')
        fold += 1
        self.store('fold', fold)
        Y_train = self.load('Y_train')
        x_train, y_train, test_dataset, test_indices = _x[train_index], _y[train_index], (_x[test_index], _y[test_index]),Y_train.index[test_index]
        self.store('test_dataset', test_dataset)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.10,
                                              random_state=fold,
                                              shuffle=True)
        self.store('x_train', x_train)
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
            callbacks=callbacks, verbose=1)
        self.send_data_to_coordinator(model, send_to_self=True)
        iteration = self.load('iteration')
        iteration += 1
        self.store("iteration", iteration)
        self.store('kf', kf)
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

    def evaluate(self, model):
        for missing_perc in [0.05, 0.1, 0.2]:
            save_name = f"/mnt/output/Chr.22.DELS/STI/preds_mixed_mr_{missing_perc}_rs_{fold}_.csv"
            avg_accuracy = []
            preds = []
            true_labels = []
            test_dataset = self.load('test_dataset')
            test_X_missing = np.empty((test_dataset[0].shape[0] * 2, test_dataset[0].shape[1]), dtype=test_dataset[0].dtype)
            predict_onehots = model.predict(test_X_missing, verbose=0)
            map_values_1_vec = np.vectorize(map_values_1)
            map_values_2_vec = np.vectorize(map_values_2)
            test_X_missing[0::2] = map_values_1_vec(test_dataset[0])
            test_X_missing[1::2] = map_values_2_vec(test_dataset[0])
            test_X_missing = to_categorical(test_X_missing, 3)
            fold = self.load('fold')
            x_train = self.load('x_train')
            bin_labels = self.load('bin_labels')
            for i in tqdm(range(test_dataset[0].shape[0])):
                missing_index, _ = train_test_split(np.arange(x_train.shape[1]), train_size=missing_perc,
                                                    random_state=i + fold, shuffle=True, stratify=bin_labels)
                test_X_missing[i*2:i*2+2, missing_index, :] = [0, 0, 1]
            predict_onehots = model.predict(test_X_missing, verbose=0)
            for i in tqdm(range(test_dataset[0].shape[0])):
                missing_index, _ = train_test_split(np.arange(x_train.shape[1]), train_size=missing_perc, random_state=i + fold,
                                                shuffle=True, stratify=bin_labels)
                predict_missing_onehot = predict_onehots[i*2:(i+1)*2, missing_index, :]
                predict_missing = np.argmax(predict_missing_onehot, axis=2)
                predict_missing_final = np.zeros((1, predict_missing.shape[1]))
                for j in range(predict_missing.shape[1]):
                    if predict_missing[:, j].tolist() == [0, 0]:
                        predict_missing_final[:, j] = 0
                    elif predict_missing[:, j].tolist() == [0, 1]:
                        predict_missing_final[:, j] = 1
                    elif predict_missing[:, j].tolist() == [1, 0]:
                        predict_missing_final[:, j] = 2
                    elif predict_missing[:, j].tolist() == [1, 1]:
                        predict_missing_final[:, j] = 3
                    else:
                        predict_missing_final[:, j] = 4
                preds.extend(predict_missing_final.ravel().tolist())
                predict_haplotypes = np.argmax(predict_onehots[i*2:(i+1)*2], axis=2)
                for j in range(predict_onehots.shape[1]):
                    if predict_haplotypes[:, j].tolist() == [0,0]:
                        to_save_array[i, j] = '0|0'
                    elif predict_haplotypes[:, j].tolist() == [0,1]:
                        to_save_array[i, j] = '0|1'
                    elif predict_haplotypes[:, j].tolist() == [1,0]:
                        to_save_array[i, j] = '1|0'
                    elif predict_haplotypes[:, j].tolist() == [1, 1]:
                        to_save_array[i, j] = '1|1'
                    else:
                        to_save_array[i, j] = '.|.'
                label_missing_onehot = test_dataset[0][i:i + 1, missing_index]
                label_missing = test_dataset[0][i:i + 1, missing_index]
                true_labels.extend(label_missing.ravel().tolist())
                correct_prediction = np.equal(predict_missing_final, label_missing)
                accuracy = np.mean(correct_prediction)
                avg_accuracy.append(accuracy)
            Y_train = self.load('Y_train')
            headers = self.load('headers')
            df = pd.DataFrame(to_save_array, columns= headers[:], index = Y_train.index[test_index])
            df.to_csv(save_name)
            log_msg = 'The average imputation accuracy on test data with {} missing genotypes is {:.4f}: '.format(missing_perc, np.mean(avg_accuracy))
            self.log(log_msg)
            cnf_matrix = confusion_matrix(true_labels, preds)
            FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
            FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            TP = np.diag(cnf_matrix)
            TN = cnf_matrix.sum() - (FP + FN + TP)
            FP = FP.astype(float)
            FN = FN.astype(float)
            TP = TP.astype(float)
            TN = TN.astype(float)
            TPR = TP/(TP+FN)
            TNR = TN/(TN+FP)
            accuracies = self.load('accuracies')
            accuracies.append(np.mean(avg_accuracy))
            self.save('accuracies', accuracies)


    def run(self):
        self.log("Start aggregation")
        data = self.gather_data()
        model = data[0]
        model.layers[0].set_weights(self.aggregate_weights(data, 0))
        model.layers[-1].set_weights(self.aggregate_weights(data, -1))
        self.evaluate(model)
        iteration = self.load("iteration")
        N_SPLITS = self.load('N_SPLITS')
        if iteration >= N_SPLITS:
            model.save('mnt/output/trained_model')
            return 'terminal'
        else:
            self.broadcast_data(model, send_to_self = True)
            return 'obtain_weights'

@app_state('write')
class WriteState(AppState):

    def register(self):
        self.register_transition('terminal')

    def run(self):
        self.log("Write results")
        return 'terminal'
