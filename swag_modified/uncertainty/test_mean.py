
#%%

import numpy as np

matrix = np.arange(0, 100)
matrix = np.vstack([matrix, matrix, matrix])
np.mean(matrix, axis=1)
np.std(matrix, axis=1)
# %%

np.mean(matrix, axis=1).shape
# %%
