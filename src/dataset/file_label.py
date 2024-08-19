import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold


class FileLabel:
    def __init__(
            self,
            label_file: str,
            data_dir: str,
            antibiotic: str
    ):
        self.label_file = label_file
        self.data_dir = data_dir
        self.antibiotic = antibiotic
        self.label_lookup = self._load_label_lookup()

    def _load_label_lookup(self):
        data = pd.read_csv(self.label_file, dtype=str).iloc
        data['files'] = self.data_dir + data['file_name']
        data.dropna(subset=[self.antibiotic], inplace=True)
        filtered = data[['files', self.antibiotic, 'genome_id']].dropna()
        return dict(zip(filtered['files'], zip(filtered[self.antibiotic], filtered['genome_id'])))

    def get_train_test_path(self, test_size=0.2, random_state=38):
        """
        To get the path and labels, so they can be loaded later
        :param test_size:
        :param random_state:
        :return:
        """
        files = list(self.label_lookup.keys())
        labels = list(self.label_lookup.values())

        train_files, test_files, train_labels, test_labels = train_test_split(
            files,
            labels,
            test_size=test_size,
            random_state=random_state
        )

        return train_files, test_files, train_labels, test_labels

    def get_k_fold_train_test_path(self, n_splits, random_state=38):
        """
        To get the path and labels, so they can be loaded later
        :param n_splits:
        :param random_state:
        :return:
        """
        files = list(self.label_lookup.keys())
        labels = list(self.label_lookup.values())

        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        for train_index, test_index in kf.split(files):
            train_files = [files[i] for i in train_index]
            test_files = [files[i] for i in test_index]
            train_labels = [labels[i] for i in train_index]
            test_labels = [labels[i] for i in test_index]
            yield train_files, test_files, train_labels, test_labels


if __name__ == '__main__':
    file_label = FileLabel(
        '../../../volatile/cgr_labels/cgr_label.csv',
        '../../../volatile/cgr/',
        'mic_AMK'
    )
    train_files = list(file_label.get_k_fold_train_test_path(10))
    print()
