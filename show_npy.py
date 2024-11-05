import numpy as np
import matplotlib.pyplot as plt

def visualize_npy(npy_file):
    # Load the .npy file
    data = np.load(npy_file)
    print (np.unique(data))
    # Plotting the data
    plt.imshow(data*255, cmap='viridis')  # You can change the colormap as per your preference
    plt.colorbar()  # Add a colorbar to show the scale
    plt.title('Visualization of .npy data')
    plt.savefig('1.png')

if __name__ == "__main__":
    # npy_file = 'masks/HCM/20170307 Gu Jing Jing/1.npy'  # Replace 'your_file.npy' with the path to your .npy file
    npy_file = 'image_npy/HCM/20200720 zhangshanshan/1.npy'  # Replace 'your_file.npy' with the path to your .npy file
    visualize_npy(npy_file)
