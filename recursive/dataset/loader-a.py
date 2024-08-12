import numpy as np

from recursive.dataset.file_label import FileLabel
from recursive.genome.sequence import Sequence
from tqdm import tqdm
from recursive.log import logger
from multiprocessing import Pool
from recursive.features import lookup


class Loader:
    def __init__(
            self,
            file_label: FileLabel,
            num_workers=8
    ):
        self.file_label = file_label
        self.num_workers = num_workers

        self.train_files = None
        self.test_files = None
        self.train_labels = None
        self.test_labels = None

        self._load_sequences()
        self.train_sequences = []
        self.test_sequences = []
        self._get_train_seq()
        self._get_test_seq()

    def _load_sequences(self):
        self.train_files, self.test_files, self.train_labels, self.test_labels = self.file_label.get_train_test_path()

    @staticmethod
    def _create_one_sequence(args):
        file = args
        return Sequence(file)

    def _get_train_seq(self):
        logger.info('Loading training sequences...')
        with Pool(self.num_workers) as p:
            self.train_sequences = list(tqdm(p.imap(
                Loader._create_one_sequence,
                self.train_files
            ), total=len(self.train_files)))

    def _get_test_seq(self):
        logger.info('Loading test sequences...')
        with Pool(self.num_workers) as p:
            self.test_sequences = list(tqdm(p.imap(
                Loader._create_one_sequence,
                self.test_files
            ), total=len(self.test_files)))

    @staticmethod
    def _get_one_kmer_dataset(args):
        seq, k = args
        return seq.get_kmer_count(k)

    def get_kmer_dataset(self, k: int, batch_size=80):
        logger.info(f'Getting k-mer dataset for k={k}...')

        train_kmer = []
        test_kmer = []

        for i in range(0, len(self.train_sequences), batch_size):
            batch_sequences = self.train_sequences[i:i + batch_size]
            with Pool(self.num_workers) as p:
                train_kmer.extend(tqdm(p.imap(
                    Loader._get_one_kmer_dataset,
                    [(seq, k) for seq in batch_sequences]
                ), total=len(batch_sequences)))

        for i in range(0, len(self.test_sequences), batch_size):
            batch_sequences = self.test_sequences[i:i + batch_size]
            with Pool(self.num_workers) as p:
                test_kmer.extend(tqdm(p.imap(
                    Loader._get_one_kmer_dataset,
                    [(seq, k) for seq in batch_sequences]
                ), total=len(batch_sequences)))

        return (
            np.asarray(train_kmer, dtype=np.float32),
            np.asarray(test_kmer, dtype=np.float32),
            np.asarray(self.train_labels, dtype=np.float32),
            np.asarray(self.test_labels, dtype=np.float32)
        )

    @staticmethod
    def _get_one_extended_dataset(args):
        seq = args
        return seq.get_count_from_lookup()

    def get_extended_dataset(self):
        """
        Get the extended dataset for the training and test sequences
        :return: tuple: The training and test datasets
        """
        logger.info(f'Getting extended dataset...')
        with Pool(self.num_workers) as p:
            train_ext = list(tqdm(p.imap(
                Loader._get_one_extended_dataset,
                self.train_sequences
            ), total=len(self.train_sequences)))
            test_ext = list(tqdm(p.imap(
                Loader._get_one_extended_dataset,
                self.test_sequences
            ), total=len(self.test_sequences)))

        return (
            np.asarray(train_ext, dtype=np.float32),
            np.asarray(test_ext, dtype=np.float32),
            np.asarray(self.train_labels, dtype=np.float32),
            np.asarray(self.test_labels, dtype=np.float32)
        )

