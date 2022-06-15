import pandas as pd

# check if download was ok!
try:
    raw_data = pd.read_hdf("test.h5", "table").iloc[:, :].values
    if raw_data.shape != (404000, 806):
        print(f"FAILED: test data shape should be (404000, 806), but it was {raw_data.shape}")
    else:
        print(f"PASSED: Download of test data worked")
except:
    print(f"No 'test.h5' was found!")
    
try:
    raw_data = pd.read_hdf("val.h5", "table").iloc[:, :].values
    if raw_data.shape != (403000, 806):
        print(f"FAILED: validation data shape should be (403000, 806), but it was {raw_data.shape}")
    else:
        print(f"PASSED: Download of validation data worked")
except:
    print(f"FAILED: No 'val.h5' was found!")
    
# Only comment out if you have time (takes a while to load the train data)   
# try:
#     raw_data = pd.read_hdf("train.h5", "table").iloc[:, :].values
#     if raw_data.shape != (1211000, 806):
#         print(f"FAILED: validation data shape should be (1211000, 806), but it was {raw_data.shape}")
#     else:
#         print(f"PASSED: Download of train data worked")
# except:
#     print(f"FAILED: No 'train.h5' was found!")