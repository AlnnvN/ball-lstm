import numpy as np

array = np.load("ball_dataset.npy", allow_pickle=True)
print(array[0][0])
# for motion in array:
    # print(motion[10])