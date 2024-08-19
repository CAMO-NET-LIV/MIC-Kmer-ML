import numpy as np

from src.dataset.file_label import FileLabel
from tqdm import tqdm

from src.genome import seq_manager
from src.genome.sequence import Sequence
import ray


class Loader:
    def __init__(
            self,
            file_label: FileLabel,
            n_fold: int = 0
    ):
        self.file_label = file_label
        self.n_fold = n_fold

        self.train_files = []
        self.test_files = []
        self.train_labels = []
        self.test_labels = []

        self._load_sequence_files()

    def _load_sequence_files(self):
        if self.n_fold:
            for train_files, test_files, train_labels, test_labels in self.file_label.get_k_fold_train_test_path(self.n_fold):
                self.train_files.append(train_files)
                self.test_files.append(test_files)
                self.train_labels.append(train_labels)
                self.test_labels.append(test_labels)
        else:
            train_files, test_files, train_labels, test_labels = self.file_label.get_train_test_path()
            self.train_files.append(train_files)
            self.test_files.append(test_files)
            self.train_labels.append(train_labels)
            self.test_labels.append(test_labels)

    @staticmethod
    @ray.remote
    def _get_one_sequence(file):
        return Sequence(file)

    def _get_train_seq(self, fold):
        print('Loading training sequences...')
        train_sequences = [Loader._get_one_sequence.remote(file) for file in self.train_files[fold]]
        train_sequences = [ray.get(a) for a in tqdm(train_sequences)]
        seq_manager.add_train_sequences(train_sequences)

    def _get_test_seq(self, fold):
        print('Loading test sequences...')
        test_sequences = [Loader._get_one_sequence.remote(file) for file in self.test_files[fold]]
        test_sequences = [ray.get(a) for a in tqdm(test_sequences)]
        seq_manager.add_test_sequences(test_sequences)

    @staticmethod
    @ray.remote
    def _get_one_kmer_dataset(args):
        seq, k = args
        return seq.get_kmer_count(k)

    def get_kmer_dataset(self, k: int):
        for i in range(len(self.train_files)):
            seq_manager.clear()
            self._get_train_seq(i)
            self._get_test_seq(i)

            print(f'Getting k-mer dataset for k={k}...')

            train_kmer = [Loader._get_one_kmer_dataset.remote((seq, k)) for seq in seq_manager.train_sequences]
            test_kmer = [Loader._get_one_kmer_dataset.remote((seq, k)) for seq in seq_manager.test_sequences]

            if self.n_fold:
                yield (
                    np.asarray([ray.get(a) for a in tqdm(train_kmer)], dtype=np.float32),
                    np.asarray([ray.get(b) for b in tqdm(test_kmer)], dtype=np.float32),
                    np.asarray([label[0] for label in self.train_labels[i]], dtype=np.float32),
                    np.asarray([label[0] for label in self.test_labels[i]], dtype=np.float32),
                    [np.asarray(label[1]) for label in self.train_labels[i]],
                    [np.asarray(label[1]) for label in self.test_labels[i]]
                )

            else:
                return (
                    np.asarray([ray.get(a) for a in tqdm(train_kmer)], dtype=np.float32),
                    np.asarray([ray.get(b) for b in tqdm(test_kmer)], dtype=np.float32),
                    np.asarray([label[0] for label in self.train_labels[0]], dtype=np.float32),
                    np.asarray([label[0] for label in self.test_labels[0]], dtype=np.float32),
                    [np.asarray(label[1]) for label in self.train_labels[0]],
                    [np.asarray(label[1]) for label in self.test_labels[0]]
                )


if __name__ == '__main__':
    file_label = FileLabel(
        '../../../volatile/cgr_labels/cgr_label.csv',
        '../../../volatile/cgr/',
        'mic_AMK'
    )
    loader = Loader(file_label, 10)
    for train_kmer, test_kmer, train_label, test_label, train_genome_id, test_genome_id in loader.get_kmer_dataset(4):
        print(train_kmer.shape, test_kmer.shape, train_label.shape, test_label.shape, len(train_genome_id), len(test_genome_id))
        print()