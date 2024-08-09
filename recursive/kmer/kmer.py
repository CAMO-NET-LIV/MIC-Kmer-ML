import numpy as np

class Kmer:
    def __init__(
            self,
            k,
            keep_broken_nucleotides=False
    ):
        """
        seqs: a list of DNA sequences
        k: the "k" in k-mer
        """
        self.k = k
        self.letters = {'a': 0, 't': 1, 'g': 2, 'c': 3}
        # the multiplying number for each digit position in the k-number system
        self.multiplyBy = 4 ** np.arange(k - 1, -1, -1)
        self.n = 4 ** k  # number of possible k-mers