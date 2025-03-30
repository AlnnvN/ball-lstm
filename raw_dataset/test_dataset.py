from matplotlib import pyplot as plt
import numpy as np

array = np.load("ball_dataset_train_validation.npy", allow_pickle=True)

np.random.shuffle(array)

print(f'dataset motion size -> {len(array)}')

for motion in array:
    x_vals = [pos[0][0] for pos in motion]  
    z_vals = np.array([pos[0][2] for pos in motion])

    z_vals = np.where(z_vals < 0, 0, z_vals)  

    x_vals = np.round(x_vals, 3)
    z_vals = np.round(z_vals, 3)

    colors = ['r' if pos[1] else 'b' for pos in motion]  

    plt.figure(figsize=(6, 6))
    
    for i in range(len(x_vals)):
        plt.scatter(x_vals[i], z_vals[i], color=colors[i], s=30) 

    plt.plot(x_vals, z_vals, 'k-', alpha=0.5)  
    
    plt.grid(True)
    plt.show()
