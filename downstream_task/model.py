import sklearn.model_selection
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
import numpy as np
import sklearn
import sklearn.svm
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

class Downstream_Classifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, t_enc_out_channels, imf_enc_out_channels, enc_t, enc_imfs, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        # Data size after time-domain data encoding
        self.out_channels = t_enc_out_channels
        # Data size after frequency-domain data encoding
        self.imf_out_channels = imf_enc_out_channels
        self.classifier = sklearn.svm.SVC()
        self.enc_t = enc_t # Time domain encoder
        self.enc_imfs = enc_imfs  # Frequency domain encoder

    def fit_classifier(self, features, y):
        nb_classes = np.shape(np.unique(y, return_counts=True)[1])[0]
        train_size = np.shape(features)[0]
        self.classifier = sklearn.svm.SVC(C=10000.0, gamma='scale', probability=True)
        # SVC parameter
        param_grid = {
            'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
            'kernel': ['rbf'],
            'degree': [3],
            'gamma': ['scale'],
            'coef0': [0],
            'shrinking': [True],
            'probability': [True],
            'tol': [0.001],
            'cache_size': [200],
            # 'class_weight': [None],
            'verbose': [False],
            'max_iter': [10000000],
            'decision_function_shape': ['ovr'],
            'random_state': [None]
        }
        if train_size // nb_classes < 5 or train_size < 50:
            self.classifier.fit(features, y)
        else:
            grid_search = sklearn.model_selection.GridSearchCV(self.classifier, param_grid=param_grid, cv=5, n_jobs=5)
            if np.shape(features)[0] <= 10000:
                grid_search.fit(features, y)
            else:
                split = sklearn.model_selection.train_test_split(features, y, train_size=10000, random_state=0,
                                                                 stratify=y)
                grid_search.fit(split[0], split[2])
            self.classifier = grid_search.best_estimator_
        return self.classifier

    def encode(self, X, X_imfs, batch_size=50):
        # Data encoding
        list_dataset = []
        for i in range(len(self.enc_imfs) + 1):
            if i == 0:
                list_dataset.append(torch.from_numpy(X))
            else:
                list_dataset.append(torch.from_numpy(X_imfs[i - 1]))
        dataset = torch.utils.data.TensorDataset(*list_dataset)
        test_generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        # Size after initializing time-domain data encoding
        features_t = np.zeros((np.shape(X)[0], self.out_channels))
        # Size after initializing frequency-domain data encoding
        features_imfs = np.zeros((np.shape(X)[0], self.imf_out_channels * len(self.enc_imfs)))
        count = 0
        with torch.no_grad():
            for batch_all in test_generator:
                for i in range(len(self.enc_imfs) + 1):
                    batch_all[i] = batch_all[i].to(self.device)
                    if i == 0:
                        features_t[count * batch_size: (count + 1) * batch_size, :] = self.enc_t(
                            batch_all[i].float()).cpu().numpy()
                    else:
                        features_imfs[count * batch_size: (count + 1) * batch_size,
                        (i - 1) * self.imf_out_channels: i * self.imf_out_channels] = self.enc_imfs[i - 1](
                            batch_all[i].float()).cpu().numpy()
                count += 1
        self.enc_t = self.enc_t.train()
        for i in range(len(self.enc_imfs)):
            self.enc_imfs[i] = self.enc_imfs[i].train()
        return np.hstack((features_t, features_imfs))

    def fit(self, X, X_imfs, y, cla_path, dataname):  
        # SVM classifier training
        # Data encoding
        features = self.encode(X, X_imfs)   
        # Training classifier
        self.classifier = self.fit_classifier(features, y)
        # if not os.path.exists(cla_path):
        #     os.makedirs(cla_path)
        # torch.save(self.classifier.state_dict(), os.path.join(cla_path, dataname + '_classifier%s.pth'))
        return self

    def test(self, X, X_imfs, y, batch_size=50):
        # testing
        # Data encoding
        features = self.encode(X, X_imfs, batch_size=batch_size)
        # Calculation indicators：acc、f1、precision and recall
        acc = self.classifier.score(features, y)
        predicted = self.classifier.predict(features)
        f1 = f1_score(y, predicted, average='macro')
        precision = precision_score(y, predicted, average='macro')
        recall = recall_score(y, predicted, average='macro')
        return acc, f1, precision, recall