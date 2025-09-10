from pandas import DataFrame
from typing import Callable

from src.common.balancing import BalancedSampler

def balance_func_factory(df: DataFrame) -> Callable:

    sampler = BalancedSampler(df['label'].unique(), verbose=False)
    
    #Use technique column as samples, match by comparing values
    sampler.count_classes(list(df['label']), lambda technique, cls: technique == cls)

    return lambda _: df.iloc[sampler.next_balanced()]