import numpy as np


def gradient_kernel(
        vox=[1.0,1.0,1.0],
    ):

    # Gradient kernels
    k_grad_x = np.zeros((3,1,1))
    k_grad_x[0,0,0], k_grad_x[2,0,0] = 1, -1
    k_grad_x /= 2*vox[0]
    
    k_grad_y = np.zeros((1,3,1))
    k_grad_y[0,0,0], k_grad_y[0,2,0] = 1, -1
    k_grad_y /= 2*vox[1]
    
    k_grad_z = np.zeros((1,1,3))
    k_grad_z[0,0,0], k_grad_z[0,0,2] = 1, -1
    k_grad_z /= 2*vox[2]
    
    return k_grad_x, k_grad_y, k_grad_z
    
    
def laplacian_kernel(
        vox=[1.0,1.0,1.0],
    ):
    
    # Laplacian kernels
    k_del2_x = np.zeros((3,1,1))
    k_del2_x[0,0,0], k_del2_x[1,0,0], k_del2_x[2,0,0] = 1, -2, 1
    k_del2_x /= vox[0]**2
    
    k_del2_y = np.zeros((1,3,1))
    k_del2_y[0,0,0], k_del2_y[0,1,0], k_del2_y[0,2,0] = 1, -2, 1
    k_del2_y /= vox[1]**2
    
    k_del2_z = np.zeros((1,1,3))
    k_del2_z[0,0,0], k_del2_z[0,0,1], k_del2_z[0,0,2] = 1, -2, 1
    k_del2_z /= vox[2]**2
    
    return k_del2_x, k_del2_y, k_del2_z