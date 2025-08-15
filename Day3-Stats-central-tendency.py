import numpy as np
from scipy import stats
import math

# Test scores of students
scores = [75, 85, 90, 95, 85, 80, 78, 85, 92, 88]

# mean
mean_score = np.mean(scores)
print("Mean score:", mean_score)

# median
median_score = np.median(scores)
print("Median score:", median_score)

# mode
mode_score = stats.mode(scores)
print("Mode:", mode_score[0])

# variance
variance_score = np.var(scores)
print("variance score:", variance_score)

# standard deviation - square root of variance
standard_dev_score = np.std(scores)
print("standard deviation:", standard_dev_score, math.sqrt(variance_score))
