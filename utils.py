import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm
import pandas as pd

def map_values_1(self, x):
    return 0 if (x == 0 or x == 1) else 1

def map_values_2(self, x):
    return 0 if (x == 0 or x == 2) else 1

def evaluate_model(instance, model, mode, test_dataset, test_Y):
    instance.log("EVALUATION")
    for missing_perc in [0.05, 0.1, 0.2]:
        instance.log(mode)
        save_name = f"/mnt/output/preds_mixed_mr_{missing_perc}_{mode}.csv"
        # save_name_numpy = f"/mnt/output/preds_mixed_mr_{missing_perc}_rs"
        avg_accuracy = []
        preds = []
        true_labels = []
        #test_dataset = self.load('test_dataset')
        to_save_array = np.zeros((test_dataset[0].shape[0], test_dataset[0].shape[1]), dtype=object)
        test_X_missing = np.empty((test_dataset[0].shape[0] * 2, test_dataset[0].shape[1]), dtype=test_dataset[0].dtype)
        map_values_1_vec = np.vectorize(instance.map_values_1)
        map_values_2_vec = np.vectorize(instance.map_values_2)
        test_X_missing[0::2] = map_values_1_vec(test_dataset[0])
        test_X_missing[1::2] = map_values_2_vec(test_dataset[0])
        test_X_missing = to_categorical(test_X_missing, 3)
        # x_train = self.load('x_train')
        for i in tqdm(range(test_dataset[0].shape[0])):
            missing_index, _ = train_test_split(np.arange(test_dataset[0].shape[1]), train_size=missing_perc,
                                                   random_state=i, shuffle=True)
            test_X_missing[i*2:i*2+2, missing_index, :] = [0, 0, 1]
        predict_onehots = model.predict(test_X_missing, verbose=0, batch_size=5)
            #np.save(save_name_numpy, predict_onehots, allow_pickle=False)
        for i in tqdm(range(test_dataset[0].shape[0])):
            missing_index, _ = train_test_split(np.arange(test_dataset[0].shape[1]), train_size=missing_perc, random_state=i,
                                                shuffle=True)
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
            #Y_train = self.load('Y_train')
            #headers = self.load('headers')
            #test_index = self.load('test_index')
            #df = pd.DataFrame(to_save_array, columns= headers[:], index = Y_train.index[test_index])
            #test_Y = self.load('test_Y')
        #df = pd.DataFrame(to_save_array, columns= headers[:], index = test_Y)
        df = pd.DataFrame(to_save_array, index = test_Y)
        df.to_csv(save_name)
        log_msg = 'The average imputation accuracy on test data with {} missing genotypes is {:.4f}. '.format(missing_perc, np.mean(avg_accuracy))
        instance.log(log_msg)
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
        instance.log(f"Sensitivity: {np.mean(TPR)}")
        instance.log(f"Specificity: {np.mean(TNR)}")
        instance.log(f"F1-score macro: {f1_score(true_labels, preds, average='macro')}")
        instance.log(f"F1-score micro: {f1_score(true_labels, preds, average='micro')}")
        instance.log(f"Average accuracy: {np.mean(avg_accuracy)}")

