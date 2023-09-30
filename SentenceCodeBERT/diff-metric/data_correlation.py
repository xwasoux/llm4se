import numpy as np

def calculate_correlation(series1, series2):
    if len(series1) != len(series2):
        raise ValueError("Both series must have the same length.")
    correlation_coefficient = np.corrcoef(series1, series2)[0, 1]
    
    return correlation_coefficient

