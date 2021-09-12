


from pylab import *
import operator
import networkx as nx
import gudhi as gd
from sklearn.metrics import pairwise_distances as dists
import numpy as np
import math
import pandas as pd







def positions(points):
    positions = {}
    for i, pt in enumerate(points):
        positions[i] = pt
    return positions










def dist2PD(distance_matrix,max_dim=2,max_edge_length=1):

    rips_complex = gd.RipsComplex(distance_matrix = distance_matrix,max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)        
    diag = simplex_tree.persistence()
    diag = [diag[i] for i in where([s[0]<max_dim for s in diag])[0]]
    return simplex_tree,diag



       


def max_matrix(M,m,n):
    K=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            K[i][j]=max(M[i][j],M[j][i])
    return (K)


########################################
########################################
######################################## 
####          copied over
########################################
########################################
########################################





def get_filtration_matrix(XX,max_value=.1,numsteps=8):
    
    #symmetrize input matrix
    m,n = shape(XX)
    XX = max_matrix(XX,m,n)
    
    #filtration_matrix = (XX==0)*np.max(XX)*10 + XX #make non-edges some huge number
    filtration_matrix = (XX==0)*max_value + XX #make non-edges some huge number
    filtration_matrix = filtration_matrix  - np.diag(np.diag(filtration_matrix))
    filtration_steps =  np.linspace(0,np.max(XX),numsteps)
    return filtration_matrix,filtration_steps



def get_weighted_filtration(G, # original graph
                            filtration_matrix,#matrix to filter 
                            filtration_steps):#steps for filtering

    labels = list(G.nodes())#node names
    
    GG = G
    filtration=[]
    for k,step in enumerate(filtration_steps):
        AA = np.array( filtration_matrix <= step,int)
        for i in range(len(AA)):
            AA[i,i] = 0
        A2 = pd.DataFrame(AA, index=labels, columns=labels)    
        GG = nx.to_networkx_graph(A2)
        #mapping = dict(zip(GG, string.ascii_lowercase))
        #GG = nx.relabel_nodes(G, mapping)  # nodes are characters a through z
        filtration.append(GG)
    return filtration

 



def get_filtration_matrix_3(XX,max_value=.1,numsteps=8):
    
    #symmetrize input matrix
    m,n = shape(XX)
    XX = max_matrix(XX,m,n)
    
    fil_matrix = max_value - XX
    fil_matrix = fil_matrix - np.diag(np.diag(fil_matrix))
    
    #    return fil_matrix, max_value - np.linspace(0,max_value,numsteps)
    filtration_steps = max_value - np.linspace(.003,max_value*.95,numsteps)
    return fil_matrix, filtration_steps

def get_filtration_matrix_2(XX,max_value=.1,numsteps=8):
    
    #symmetrize input matrix
    m,n = shape(XX)
    XX = max_matrix(XX,m,n)
    
    fil_matrix = max_value - XX
    fil_matrix = fil_matrix - np.diag(np.diag(fil_matrix))
    
    return fil_matrix, max_value - np.linspace(.00000001,max_value*.9,numsteps)



def get_1_cycles(diag): # return numpy array of 1-cycles from persistence diagram
    one_cycles = []
    for d in diag:
        if d[0]==1:
            one_cycles.append(([d[1][0],d[1][1]])  )
    return np.array(one_cycles).T





