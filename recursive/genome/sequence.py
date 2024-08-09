import numpy as np
import time


class Sequence:
    def __init__(
            self,
            filepath: str,
            keep_read_error=False
    ):
        self.filepath = filepath
        self.keep_read_error = keep_read_error
        self._sequence = self._read_sequence()

    def __len__(self):
        return len(self._sequence)

    def __getitem__(self, index):
        return self._sequence[index]

    def __str__(self):
        return self._sequence

    def _read_sequence(self):
        with open(self.filepath, 'r') as f:
            string = f.read().split('\n')

        # use filter() to remove header and empty lines
        string = list(filter(lambda x: not x.startswith('>') and x != '', string))

        # join the list of strings into one string
        string = ''.join(string)

        # change to lower case
        string = string.lower()

        if self.keep_read_error:
            # change any character other than 'a', 't', 'g', 'c' to 'n'
            string = ''.join([c if c in 'atgc' else 'n' for c in string])
        else:
            # remove any character other than 'a', 't', 'g', 'c'
            string = ''.join([c for c in string if c in 'atgc'])

        return string

    def get_kmer_count(
            self,
            k: int
    ):
        """
        Given a kmer sequence, return the transition frequency matrix.
        """
        nuc_map = {'a': 0, 't': 1, 'g': 2, 'c': 3, 'n': 4} if self.keep_read_error else {'a': 0, 't': 1, 'g': 2, 'c': 3}

        base = 5 if self.keep_read_error else 4

        multiply_by = base ** np.arange(k - 1, -1, -1)
        n = base ** k  # number of possible k-mers

        def kmer_mapping(i):
            return np.dot([nuc_map[c] for c in self[i:i + k]], multiply_by)

        kmer_seq = list(map(kmer_mapping, range(len(self) - k + 1)))
        kmer_seq = np.array(kmer_seq, dtype=np.int32)

        kmer_count = np.bincount(kmer_seq, minlength=n)

        return kmer_count


if __name__ == '__main__':
    start = time.time()
    for _ in range(1, 5):
        seq = Sequence('/home/yinzheng/Documents/pycharm/volatile/e_coli_mic/562.5419.fna', keep_read_error=True)
        print(len(seq))
        kc = seq.get_kmer_count(8)

    print(f'Time: {time.time() - start}')
