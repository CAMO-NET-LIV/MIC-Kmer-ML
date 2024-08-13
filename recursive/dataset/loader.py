import numpy as np

from recursive.dataset.file_label import FileLabel
from tqdm import tqdm

from recursive.genome.sequence_manager import SequenceManager
from recursive.log import logger
from multiprocessing import Pool
from recursive.genome.sequence import Sequence
from multiprocessing import Manager


class Loader:
    def __init__(
            self,
            file_label: FileLabel,
            num_workers=8
    ):
        self.file_label = file_label
        self.num_workers = num_workers

        # Initialize a Manager to manage shared objects
        manager = Manager()

        # Initialize seq_manager with managed lists
        global seq_manager
        seq_manager = SequenceManager(manager)

        self.train_files = None
        self.test_files = None
        self.train_labels = None
        self.test_labels = None

        self._load_sequence_files()
        self._get_train_seq()
        self._get_test_seq()

    def _load_sequence_files(self):
        self.train_files, self.test_files, self.train_labels, self.test_labels = self.file_label.get_train_test_path()

    def _get_train_seq(self):
        logger.info('Loading training sequences...')
        with Pool(self.num_workers) as pool:
            train_sequences = list(tqdm(pool.imap(Sequence, self.train_files), total=len(self.train_files)))

        seq_manager.add_train_sequences(train_sequences)

    def _get_test_seq(self):
        logger.info('Loading test sequences...')
        with Pool(self.num_workers) as pool:
            test_sequences = list(tqdm(pool.imap(Sequence, self.test_files), total=len(self.test_files)))

        seq_manager.add_test_sequences(test_sequences)

    @staticmethod
    def _get_one_kmer_dataset(args):
        seq, k = args
        return seq.get_kmer_count(k)

    def get_kmer_dataset(self, k: int):
        logger.info(f'Getting k-mer dataset for k={k}...')

        with Pool(self.num_workers) as p:
            train_kmer = list(tqdm(p.imap(
                Loader._get_one_kmer_dataset,
                ((seq, k) for seq in seq_manager.train_sequences)
            ), total=len(seq_manager.train_sequences)))

        with Pool(self.num_workers) as p:
            test_kmer = list(tqdm(p.imap(
                Loader._get_one_kmer_dataset,
                ((seq, k) for seq in seq_manager.test_sequences)
            ), total=len(seq_manager.test_sequences)))

        return (
            np.asarray(train_kmer, dtype=np.float32),
            np.asarray(test_kmer, dtype=np.float32),
            np.asarray(self.train_labels, dtype=np.float32),
            np.asarray(self.test_labels, dtype=np.float32)
        )

    @staticmethod
    def _get_one_extended_dataset(seq):
        return seq.get_count_from_seg_manager()

    def get_extended_dataset(self):
        """
        Get the extended dataset for the training and test sequences
        :return: tuple: The training and test datasets
        """
        logger.info(f'Getting extended dataset...')
        with Pool(self.num_workers) as p:
            train_ext = list(tqdm(p.imap(
                Loader._get_one_extended_dataset,
                seq_manager.train_sequences
            ), total=len(seq_manager.train_sequences)))
            test_ext = list(tqdm(p.imap(
                Loader._get_one_extended_dataset,
                seq_manager.test_sequences
            ), total=len(seq_manager.test_sequences)))

        return (
            np.asarray(train_ext, dtype=np.float32),
            np.asarray(test_ext, dtype=np.float32),
            np.asarray(self.train_labels, dtype=np.float32),
            np.asarray(self.test_labels, dtype=np.float32)
        )
