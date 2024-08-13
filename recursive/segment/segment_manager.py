import time
import os
from recursive.genome import km
from recursive.log import logger
from recursive.etc.config import config


class SegmentManager:
    def __init__(
            self,
    ):
        """
        Constructor for the SegmentManager class
        :param kmer: int: The k-mer value
        """
        self.segments = []

    def __iter__(self):
        return iter(self.segments)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        return self.segments[index]

    def get_copy(self):
        """
        Return a copy of the segments table
        """
        return self.segments.copy()

    def add_all_kmer(self, k: int, keep_read_error=False):
        base = 5 if keep_read_error else 4
        kmers = [km.reverse_kmer_mapping(i, k) for i in range(base ** k)]
        self.add_subsequences(kmers, remove_duplicates=False)

    def use_subset(self, indices: [int]):
        """
        Filter the segments table by the keys
        """
        try:
            self.segments = [self.segments[i] for i in indices]
        except IndexError:
            logger.error('Index out of range')
            self.segments = []

        logger.info(f'Keeping {len(self.segments)} segments as shown below:\n{self.segments}')

    def save(self, filename: str):
        """
        Save the segments table to a file
        :param filename: str: The name of the file to save the lookup table
        """
        path = os.path.join(config['save_dir'], filename)

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            for item in self.segments:
                f.write("%s\n" % item)

        logger.info(f'Saved {len(self.segments)} segments to {path}')

    def load(self, filename: str):
        """
        Load the segments table from a file
        :param filename: str: The name of the file to load the lookup table
        """
        path = os.path.join(config['save_dir'], filename)

        with open(path, 'r') as f:
            self.segments = f.read().split('\n')

        logger.info(f'Loaded {len(self.segments)} segments from {path}')

    def add_subsequences(self, sequences: [str], remove_duplicates=True):
        """
        Add a list of sequences to the lookup table
        Note: This method uses set to remove duplicates, but it changes order of the sequences
        :param sequences: list: The list of sequences to add#
        :param remove_duplicates: bool: Remove duplicates from the list
        """
        self.segments = self.segments + sequences
        if remove_duplicates:
            self.segments = list(set(self.segments))

        logger.info(f'Number of segments: {len(self.segments)}')

    def segments_pruning(self, importance_ranking: [int]):
        """
        Prune the list of strings to remove substrings or master strings
        :param importance_ranking: list: The list of indices of the most important features (descending order)
        :return:
        """

        # Result list to store non-substring strings
        result = []

        # Sort the segments based on the importance ranking
        ranked_segments = [self.segments[i] for i in importance_ranking]

        # Iterate through the ranked segments and check if any segment is a substring of another
        blocked = {}
        for i in range(len(ranked_segments)):
            if i in blocked:
                continue
            master_sub = {i}
            for j in range(len(ranked_segments)):
                if j in blocked:
                    continue
                if i != j and (ranked_segments[i] in ranked_segments[j] or ranked_segments[j] in ranked_segments[i]):
                    master_sub.add(j)

            # keep the one has the highest importance
            result.append(ranked_segments[min(master_sub)])
            # block the rest
            for k in master_sub - {min(master_sub)}:
                blocked[k] = True

        # Update the segments with the pruned list
        self.segments = result

        logger.info(f'Number of segments after pruning: {len(self.segments)}')


if __name__ == '__main__':
    seg_manager = SegmentManager()
    seg_manager.segments = ['aaa', 'aa', 'aa', 'a', 'ab', 'b', 'ba', 'bac', 'c', 'ca', 'cab', 'd', 'da', 'dac', 'e', 'ea', 'eac']
    seg_manager.segments_pruning([1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    print()