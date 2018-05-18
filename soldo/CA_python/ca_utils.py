def get_densities(idx_list, victims, attackers):
    from oct2py import octave
    from scipy import sparse
    from itertools import product
    # TODO: add the right to the folder CA_python
    octave.addpath('/Users/lotm/collsec/soldo/CA_python/')
    
    import time
    import numpy as np
    
    tick = time.time()

    k, l, Nx, Ny, Qx, Qy, Dnz = octave.cross_association(idx_list+1)

    print "CA is done in {} sec".format(time.time() - tick)

    k, l = int(k), int(l)
    Qx = Qx.ravel().astype(np.int) - 1
    Qy = Qy.ravel().astype(np.int) - 1
    
    Nx = Nx.ravel()
    Ny = Ny.ravel()
    
    Dnz = Dnz.reshape((k,l)).astype(np.float)
    
    Nz_x, Nz_y = Dnz.nonzero()
    
    x_idx = {}
    y_idx = {}
    
    for i in np.unique(Nz_x):
        x_idx[i] = np.where(Qx==i)[0]
    
    for j in np.unique(Nz_y):        
        y_idx[j] = np.where(Qy==j)[0] 
    
    values = []
    inds_row = []
    inds_col = []
    for i in np.unique(Nz_x):
        for j in np.unique(Nz_y):
            #print x_idx[i], y_idx[j]
            temp = np.array( list( product(x_idx[i], y_idx[j] ) ) )  
            inds_row += list(temp[:,0])
            inds_col += list(temp[:,1])
            values += [Dnz[i,j] / (Nx[i] * Ny[j])]* temp.shape[0]
            
    sp_mat = sparse.csc_matrix((values, (inds_row,inds_col)), shape=(victims, attackers))

    return sp_mat