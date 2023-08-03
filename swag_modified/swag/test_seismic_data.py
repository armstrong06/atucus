
#%%
import seismic_data

seismic_data.loaders(
    "uuss_train.h5",
    "uuss_validation.h5",
    "../../data",
    64,
    1,
    400,
    0.5, 
    0.01, 
    1
)
#%% 