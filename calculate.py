import numpy as np

def log2_ratio(x, y):

    result = np.log2(x) - np.log2(y)

    return "{:.5e}".format(result)



print(log2_ratio(7.3412e-03, 3.1952e-03))
print(log2_ratio(3.1952e-03, 1.4952e-03))
print(log2_ratio(1.4952e-03, 7.2512e-04))
print(log2_ratio(7.2512e-04, 2.5515e-05))
print(log2_ratio(2.5515e-05, 6.3128e-06))
print(log2_ratio(1.5808e-04, 3.9371e-05))
print(log2_ratio(6.3790e-04, 4.5183e-01))
print(log2_ratio(4.5183e-01, 5.0000e-01))