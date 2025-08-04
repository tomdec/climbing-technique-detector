from typing import List
from numpy import zeros, int32

class BalancedSampler:

    @property
    def classes(self) -> List[int | str]:
        """Set of classes or labels each sample belongs to."""
        return self._classes
    
    def __init__(self, classes, verbose = False):
        self._classes = classes
        self._verbose = verbose

        self._class_counts = zeros(len(self._classes), dtype=int32)

    def count_classes(self, match_func):
        pass

    def next_balanced(self):
        
        pass