from typing import List, Any, Callable
from numpy import zeros, int32, where, random

class BalancedSampler:

    _samples_classes: List[int]
    """List of indexes of `self._classes` for each sample, constructed from the `samples` argument in `count_classes`"""

    def __init__(self, classes, verbose = False):
        self._classes = classes
        self._verbose = verbose

        self._class_counts = zeros(len(self._classes), dtype=int32)
        self._samples_classes = []

    def count_classes(self, samples: List[Any], match_func: Callable[[Any, int | str], bool]):
        """
        Counts the amount of times each class is present in the list `samples`.

        Args:
            samples: List of the samples in the dataset.
            match_func: function that takes a sample and a class as arguments, and returns a bool indicating
                if that sample belong to the given class.
        """

        for sample in samples: 
            for idx, cls in enumerate(self._classes):
                if match_func(sample, cls):
                    self._class_counts[idx] += 1
                    self._samples_classes.append(idx)
        
        if (self._verbose):
            print(dict(zip(self._classes, self._class_counts)))

        #Correction to avoid DividedByZero errors.
        self._class_counts = where(self._class_counts == 0, 1, self._class_counts)

        self._class_weights = sum(self._class_counts) / self._class_counts
        self._sample_weights = self._class_weights[self._samples_classes]

        self._sample_probabilities = self._sample_weights / sum(self._sample_weights)

    def next_balanced(self) -> int:
        """
        Return a random index for a sample, weighted according to their class. 
        So that, in general, each class is sampled an equivalent amount of times.
        """
        return random.choice(len(self._sample_probabilities), p=self._sample_probabilities)