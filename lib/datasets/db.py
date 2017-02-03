"""
Generic database class.
"""
import random
import numpy as np
import itertools


class DB(object):
    # Data (filename) keys. Used to retrieve data and labels
    set_data = {'train': [], 'test': []}

    _perm = {}

    def __init__(self):
        """Initializes sets of data and its class."""
        self.initialize_data()
        for phase in self.set_data.keys():
            self._initialize_iterations(phase)

    def initialize_data(self):
        """ Must overwrite to initialize set_data. set_data should be a
        dictionary of keys. The keys are unique identifiers for loading a datum
        and a label. Please implement get_datum and get_label as well."""
        raise NotImplementedError()

    def get_datum(self, datum):
        """Loads filename given data."""
        pass

    def get_label(self, datum):
        """Loads label given data."""
        pass

    def has_next(self, phase):
        assert phase in self.set_data.keys()
        return len(self._perm[phase]) > 0

    def get_set(self, phase, repeat=False):
        """Data pair generator. Generates a pair of data from a same set.
        Guarantees to iterate over all combinations of pairs of all set.
        If repeat is True, indefinitely iterates over all pairs.
        Yields datum, label pair.
        """
        assert phase in self.set_data.keys()
        while self.has_next(phase) or repeat:
            # If all combinations of all set has been seen, stop or initialize.
            if not self._perm[phase]:
                if not repeat:
                    raise ValueError('Should not reach this point')
                self._initialize_iterations(phase)

            # Randomly choose a set to train.
            if phase == 'train':
                datum_idx = np.random.randrange(len(self._perm[phase]))
            elif phase == 'test':  # Choose first set at test.
                datum_idx = 0
            else:
                raise RuntimeError('Unknown phase %s.' % phase)

            # Choose random pair from a given set.
            data_idx = self._perm[phase][datum_idx].pop()
            ds = (self.set_data[phase][setidx][i] for i in dataidx_pair)
            # Remove empty set.
            # Yield a pair of data along with its metadata.
            yield self.get_imgname(self.set_data[phase][data_idx]), self.get_meta(d)

    def _initialize_iterations(self, phase):
        assert phase in self.set_data.keys()
        self._perm[phase] = np.random.permutation(np.arange(len(self.set_data)))
