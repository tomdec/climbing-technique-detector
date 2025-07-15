from pandas import DataFrame
from numpy import zeros, int32, where, random
from typing import Callable

def balance_func_factory(df: DataFrame) -> Callable:
    classes = df['technique'].unique()
    
    class_counts = zeros(len(classes), dtype=int32)
    for idx, cls in enumerate(classes):
        class_counts[idx] = df['technique'].where(lambda technique: technique == cls).count()
    class_counts = where(class_counts == 0, 1, class_counts)

    print(dict(zip(classes, class_counts)))

    class_weights = sum(class_counts) / class_counts
    sample_weights = df['technique'].map(lambda x: class_weights[list(classes).index(x)])

    sample_probabilities = sample_weights / sum(sample_weights)

    return lambda _: df.iloc[random.choice(len(sample_probabilities), p=sample_probabilities)]