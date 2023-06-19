import numpy as np
import glob
import matplotlib.pyplot as plt
import torch
import textwrap
import torch.nn.functional as F

# Helper functions for overall flow

# There are two types of data (.bov + .times / .grid + .griddata)
# Training data is in the form of (.grid + .griddata)
# Generated samples are stored in (.bov + .times) format for the visualization using Vislt software


# read (.bov + .times) data and convert it to numpy array 
def timesdata_to_array(root='Sample_3/', upper_scale = 5000, lower_scale = -4000):

    file_list = glob.glob(root+'*_si.times')
    file_list.sort()
    
    stem_list = [(i.split('/')[1]).split('_si.times')[-2] for i in file_list]
    
    grids = []
    
    count = 0
    for stem in stem_list:
        count += 1
        
        if count % 100 == 0:
            print(count)
        
        grid_file = root+stem+'.times'
        
        arr_grid = np.fromfile(grid_file, dtype='float32')

        arr_grid = arr_grid.reshape(32,32,32)

        # z,y,x -> x,y,z
        arr_grid = arr_grid.swapaxes(0,2)
        
        #arr_grid = np.clip(arr_grid, lower_scale, upper_scale)
     
        grids.append(arr_grid)
    
    grids = np.stack(grids)
    
    return grids

# read (.grid + .griddata) data and convert it to numpy array
def griddata_to_array(root='Sample_3/', upper_scale = 5000, lower_scale = -4000):

    file_list = glob.glob(root+'*.griddata')

    file_list.sort()
    
    grids = []
    
    count = 0
    
    for grid_file in file_list:
        
        arr_grid = np.fromfile(grid_file, dtype='float32')

        arr_grid = arr_grid.reshape(32,32,32)

        # z,y,x -> x,y,z
        arr_grid = arr_grid.swapaxes(0,2)
        
        #arr_grid = np.clip(arr_grid, lower_scale, upper_scale)
     
        grids.append(arr_grid)
    
    
    grids = np.stack(grids)
    
    return grids

# (optional) visualize grid data using matplotlib
def visualize_grid(data, grid_idx,num_grid = 32):
    fig = plt.figure()
    
    data = data[grid_idx,:,:,:]
        
    ax = fig.add_subplot(projection='3d')
 
    x,y,z = np.mgrid[0:1:32j, 0:1:32j, 0:1:32j]
    
    ax.scatter(x, y, z, c=data, cmap='Reds', s=5, alpha=0.1)    
  
    #ax.colorbar()

# Calculate void fraction from energy grid
def Void_Fraction(grid, temperature=298):

    lx, ly, lz = grid.shape

    low_e_grid = 0

    for x in range(lx):
        for y in range(ly):
            for z in range(lz):
                if (grid[x][y][z] <= 15*temperature):
                    low_e_grid += 1
                    
    return low_e_grid/(lx*ly*lz)

# Calculate henry coefficient from energy grid
def Henry_Coeff(grid, temperature=298):

    lx, ly, lz = grid.shape

    sum = 0

    for x in range(lx):
        for y in range(ly):
            for z in range(lz):
                sum += np.exp(-grid[x][y][z]/temperature)
            
    sum /= (lx*ly*lz)
            
    return sum

# Calculate heat of adsorption from energy grid
def Heat_of_Adsorption(grid, temperature=298, converter = 0.0083):

    lx, ly, lz = grid.shape

    HoA_sum = 0
    henry_sum = 0
    
    for x in range(lx):
        for y in range(ly):
            for z in range(lz):
                henry_sum += np.exp(-grid[x][y][z]/temperature)
                HoA_sum += np.exp(-grid[x][y][z]/temperature) * grid[x][y][z]
    
    HoA = temperature - (HoA_sum / henry_sum)
    
    return HoA*converter

# write generated grids into .bov .times format for the visualization 
def write_visit_sample(data, cell= (10,10,10), save_dir = 'save_dir', stem="test"):

    # Inverse normalize cell.

    data = data.astype(np.float32)

    cell_x, cell_y, cell_z = cell

    
    grid_data = data[0,:,:,:]
    si_data = data[1,:,:,:]
    o_data = data[2,:,:,:]
    
    
    #grid_min = -4000
    #grid_max =  5000
    
    grid_data = grid_data*-4500 + 500
    
    si_data = (si_data+1)/2
    
    o_data = (o_data+1)/2
    
    
    ########### Grid data ###############
    
    grid_data = grid_data.flatten()
    

    # Make file name.
    bov = "{}/{}.bov".format(save_dir, stem)
    times = stem + ".times"

    chan,size_x,size_y,size_z = data.shape
    

    # Write header file.
    with open(bov, "w") as bovfile:
        bovfile.write(textwrap.dedent("""\
            TIME: 1.000000
            DATA_FILE: {}
            DATA_SIZE:     {} {} {}
            DATA_FORMAT: FLOAT
            VARIABLE: data
            DATA_ENDIAN: LITTLE
            CENTERING: nodal
            BRICK_ORIGIN:        0  0  0
            BRICK_SIZE:       {} {} {}""".format(
            times, size_x, size_y, size_z, cell_x,cell_y,cell_z)
        ))
    # Write times file.
    grid_data.tofile("{}/{}".format(save_dir, times))
    
    
    ########### Si data ###############
    
    si_data = si_data.flatten()
    
    # Make file name.
    stem_si = stem + '_si'
    bov = "{}/{}.bov".format(save_dir, stem_si)
    times = stem_si + ".times"

    chan,size_x,size_y,size_z = data.shape
    
    # Write header file.
    with open(bov, "w") as bovfile:
        bovfile.write(textwrap.dedent("""\
            TIME: 1.000000
            DATA_FILE: {}
            DATA_SIZE:     {} {} {}
            DATA_FORMAT: FLOAT
            VARIABLE: data
            DATA_ENDIAN: LITTLE
            CENTERING: nodal
            BRICK_ORIGIN:        0  0  0
            BRICK_SIZE:       {} {} {}""".format(
            times, size_x, size_y, size_z, cell_x,cell_y,cell_z)
        ))
    # Write times file.
    si_data.tofile("{}/{}".format(save_dir, times))    

    
    ########### O data ###############
    
    o_data = o_data.flatten()

    
    # Make file name.
    stem_o = stem + '_o'
    bov = "{}/{}.bov".format(save_dir, stem_o)
    times = stem_o + ".times"

    chan,size_x,size_y,size_z = data.shape
    
    # Write header file.
    with open(bov, "w") as bovfile:
        bovfile.write(textwrap.dedent("""\
            TIME: 1.000000
            DATA_FILE: {}
            DATA_SIZE:     {} {} {}
            DATA_FORMAT: FLOAT
            VARIABLE: data
            DATA_ENDIAN: LITTLE
            CENTERING: nodal
            BRICK_ORIGIN:        0  0  0
            BRICK_SIZE:       {} {} {}""".format(
            times, size_x, size_y, size_z, cell_x,cell_y,cell_z)
        ))
    # Write times file.
    o_data.tofile("{}/{}".format(save_dir, times))

