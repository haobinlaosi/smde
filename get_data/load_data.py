import os
import pandas as pd
import numpy as np
import math
from sktime.datasets import load_from_arff_to_dataframe
from sklearn.preprocessing import LabelEncoder
import torch


def load_ucr_dataset(data_path, data_name):
    train_file = os.path.join(data_path, data_name, data_name + "_TRAIN.tsv")
    test_file = os.path.join(data_path, data_name, data_name + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i
    train = np.expand_dims(train_array[:, 1:], 1).astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = np.expand_dims(test_array[:, 1:], 1).astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])
    if data_name not in ['AllGestureWiimoteX','AllGestureWiimoteY','AllGestureWiimoteZ','BME','Chinatown',
                       'Crop','EOGHorizontalSignal','EOGVerticalSignal','Fungi','GestureMidAirD1',
                       'GestureMidAirD2','GestureMidAirD3','GesturePebbleZ1','GesturePebbleZ2','GunPointAgeSpan',
                       'GunPointMaleVersusFemale','GunPointOldVersusYoung','HouseTwenty','InsectEPGRegularTrain','InsectEPGSmallTrain',
                       'MelbournePedestrian','PickupGestureWiimoteZ','PigAirwayPressure','PigArtPressure','PigCVP',
                       'PLAID','PowerCons','Rock','SemgHandGenderCh2','SemgHandMovementCh2',
                       'SemgHandSubjectCh2','ShakeGestureWiimoteZ','SmoothSubspace','UMD']:
        return train, train_labels, test, test_labels
    # Standardized processing
    mean = np.nanmean(np.concatenate([train, test]))
    var = np.nanvar(np.concatenate([train, test]))
    train = (train - mean) / math.sqrt(var)
    test = (test - mean) / math.sqrt(var)
    return train, train_labels, test, test_labels


def load_uea_dataset(data_path, data_name):
    train_file = os.path.join(data_path, data_name, data_name + '_TRAIN.arff')
    test_file = os.path.join(data_path, data_name, data_name + '_TEST.arff')
    # Get arff format
    train_data, train_labels = load_from_arff_to_dataframe(train_file)
    test_data, test_labels = load_from_arff_to_dataframe(test_file)

    def convert_data(data):
        # Expand the series to numpy
        data_expand = data.applymap(lambda x: x.values).values
        # Single array, then to tensor
        data_numpy = np.stack([np.vstack(x).T for x in data_expand])
        tensor_data = torch.Tensor(data_numpy)
        return tensor_data

    train_data, test_data = convert_data(train_data), convert_data(test_data)
    # Encode labels as often given as strings
    encoder = LabelEncoder().fit(train_labels)
    train_labels, test_labels = encoder.transform(train_labels), encoder.transform(test_labels)
    train_labels, test_labels = torch.Tensor(train_labels), torch.Tensor(test_labels)
    # original UEA(0,1,2) [instances, length, features/channels]
    # UEA(0,1,2) --> later will be permuted in dataloader-->get UEA(0,2,1) [instances, features/channels, length]
    train_data = np.transpose(train_data, (0,2,1))
    test_data = np.transpose(test_data, (0,2,1))
    return train_data.numpy(), train_labels.numpy(), test_data.numpy(), test_labels.numpy()