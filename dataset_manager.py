import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy import sparse
from tqdm import tqdm
from torch.utils.data import Dataset


class DatasetManager:
    def __init__(self, label_file, data_dir, transition_data_dir=None):
        self.label_file = label_file
        self.data_dir = data_dir
        self.transition_data_dir = transition_data_dir
        self.label_lookup = self._load_label_lookup()

    def _load_label_lookup(self):
        data = pd.read_csv(self.label_file, dtype=str)
        data['files'] = data['files'] + '.npz'
        return data.set_index('files').to_dict()['labels']

    def find_actual_labels(self, file_names):
        return [self.label_lookup[file] for file in file_names]

    def load_data(self):

        files = list(self.label_lookup.keys())

        labels = np.array(list(self.label_lookup.values()), dtype=np.float32)

        data_list = []
        for file in tqdm(files):
            file_path = os.path.join(self.data_dir, file)
            sparse_matrix = sparse.load_npz(file_path)

            if self.transition_data_dir:
                file_path_tran = os.path.join(self.transition_data_dir, file)
                sparse_matrix_tran = sparse.load_npz(file_path_tran).reshape(1, -1)
                concat_matrix = sparse.hstack((sparse_matrix, sparse_matrix_tran))
                data_list.append(concat_matrix)
            else:
                data_list.append(sparse_matrix)

        X = np.asarray(sparse.vstack(data_list, format='csr').toarray(), dtype=np.float32)
        y = labels
        return X, y

    def prepare_dataset(self, test_size=0.2, random_state=38, remove_columns_target_percentile=0,
                        random_transform_target_dim=0):
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        if remove_columns_target_percentile:
            X_train, X_test = self.remove_columns(remove_columns_target_percentile, X_train, X_test)

        if random_transform_target_dim:
            print(f'Random-transforming data to {random_transform_target_dim} dimensions')
            X_train, X_test = self.rand_transform(X_train, X_test, random_transform_target_dim)
            X_train, X_test = self.shift_data(X_train, X_test)

        return X_train, X_test, y_train, y_test

    def prepare_train_test_path(self, test_size=0.2, random_state=38):
        """
        To get the path and labels, so they can be loaded later
        :param test_size:
        :param random_state:
        :return:
        """
        files = list(self.label_lookup.keys())
        labels = np.array(list(self.label_lookup.values()), dtype=np.float32)

        train_files, test_files, train_labels, test_labels = train_test_split(
            files,
            labels,
            test_size=test_size,
            random_state=random_state
        )

        return train_files, test_files, train_labels, test_labels

    def remove_columns(self, percent, X_train, X_test):
        avg = X_train.mean(axis=0)
        to_remove = avg < np.percentile(avg, percent)

        X_train = self._process_data(X_train, to_remove)
        X_test = self._process_data(X_test, to_remove)

        return X_train, X_test

    def shift_data(self, X_train, X_test):
        min = X_train.min()

        X_train = X_train - min
        X_test = X_test - min
        return X_train, X_test

    def _process_data(self, data, to_remove):
        # add one column are infrequent encoding with all 0
        data = np.hstack((data, np.zeros((data.shape[0], 1), dtype=np.float32)))

        to_remove = np.hstack((to_remove, False))

        # for each row, if the removed column has any value, set the new column to 1
        removed_column_values = data[:, to_remove]
        removed_column_has_value = np.any(removed_column_values, axis=1)
        data[:, -1] = removed_column_has_value.astype(int)

        # now remove the columns
        data = data[:, ~to_remove]

        print(f'Removed {np.sum(to_remove)} / {len(to_remove)} columns')

        return data

    def rand_transform(self, X_train, X_test, target_dim):
        transform_matrix = np.random.standard_normal(size=(X_train.shape[1], target_dim))

        print('done generating random matrix')
        X_train = np.matmul(X_train, transform_matrix)
        X_test = np.matmul(X_test, transform_matrix)

        return X_train, X_test

    def prepare_k_fold_dataset(self, n_folds=5, random_state=38):
        X, y = self.load_data()
        k_fold = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
        fold_data = []
        for train_index, test_index in k_fold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            fold_data.append((X_train, X_test, y_train, y_test))
        return fold_data
