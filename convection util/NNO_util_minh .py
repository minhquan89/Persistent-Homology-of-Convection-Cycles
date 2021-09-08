


from pylab import *
import operator
import networkx as nx
import gudhi as gd
from sklearn.metrics import pairwise_distances as dists
import numpy as np
import math
#import heapq
#from operator import itemgetter





def positions(points):
    positions = {}
    for i, pt in enumerate(points):
        positions[i] = pt
    return positions

def get_filtration(points,filtration_matrix,filtration_steps):
    G = nx.Graph()
    positions = {}
    for i, pt in enumerate(points):
        positions[i] = pt
    #positions.append(pt)
    filtration=[]
    for k,step in enumerate(filtration_steps):
        A = np.array( filtration_matrix <= step,int)
        for i in range(len(A)):
            A[i,i] = 0
        G = nx.to_networkx_graph(A)
        filtration.append(G)
    return filtration

def make_filtration_fig(points,nets,titles,y_label):
    fig, axes = plt.subplots(nrows=1, ncols=len(nets),figsize=(20,4))
    ax = axes.flatten()

    for i in range(len(nets)):
        nx.draw_networkx(nets[i], pos=positions(points), ax=ax[i],s = 12,node_size=800,font_size=19,node_color='cyan')  
        ax[i].set_title(titles[i],fontsize=19)
        ax[i].tick_params(which='both',bottom=False,left=False,top=False,labelbottom=False,labelleft=False)
        ax[i].spines['bottom'].set_color('grey')
        ax[i].spines['left'].set_color('grey')
        ax[i].spines['right'].set_color('grey')
        ax[i].spines['top'].set_color('grey')
        ax[i].xaxis.label.set_color('grey')
        #ax[i].set_xlabel('xlabel', fontsize=19)
        if i==0:
             ax[i].set_ylabel(y_label,fontsize=19)
#    plt.show()
    plt.tight_layout();
    #savefig('VR_filtration.pdf');
    return




def symmetrize(r,s_type):
    if s_type == 'none':
        r=r
    if s_type == 'transpose':
        r = (r + r.T )/2
    if s_type == 'min':
        temp = np.zeros(shape(r))
        for i in range(len(r)):
            for j in range(len(r)):
                temp[i,j] = min(r[i,j],r[j,i])
        r = temp
    if s_type == 'max':
        temp = np.zeros(shape(r))
        for i in range(len(r)):
            for j in range(len(r)):
                temp[i,j] = max(r[i,j],r[j,i])
        r = temp
        
    if s_type == 'imbalance':
        r = abs(r- r.T )
    return r
        

        
def neighbor_ordering(points,s_type):
    d = dists(points,points)
    r = np.zeros(np.shape(d))
    k=[]
    for i in range(len(d)):
        r[i,:] = np.argsort(argsort(d[i,:],axis=0 ))
        b=tuple(zip(d[i,:],r[i,:]))
        b=np.array(b)
        for i in range(len(b)):
            for j in range(len(b)):
                if b[i,0]==b[j,0] and b[i,1]<b[j,1]:
                    b[j,1]=b[j,1]-1
        k.append(b[:,1])
    k=np.array(k)
    
    k = symmetrize(k,s_type)
    return k


def rank_matrix(points):
    d = dists(points,points)
    r = np.zeros(np.shape(d))
    k=[]
    for i in range(len(d)):
        r[i,:] = np.argsort(argsort(d[i,:],axis=0 ))
        b=tuple(zip(d[i,:],r[i,:]))
        b=np.array(b)
        for i in range(len(b)):
            for j in range(len(b)):
                if b[i,0]==b[j,0] and b[i,1]<b[j,1]:
                    b[j,1]=b[j,1]-1
        k.append(b[:,1])
    k=np.array(k)
    return k

def make_distance_matrix(X,m,n):
    a=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            a[i,j]=np.linalg.norm(X[j]-X[i])
    return a

def create_simplex_tree(M,max_d):
    rips_complex=gd.RipsComplex(points=M,max_edge_length=12.0)
    simplex_tree=rips_complex.create_simplex_tree(max_dimension=2)
    result_str='Rips complex is of dimension '+repr(simplex_tree.dimension())+''+\
    repr(simplex_tree.num_simplices())+' simplices-'+\
    repr(simplex_tree.num_vertices())+' vertices.'
    return simplex_tree

def plot_persistence_diagram(M,dims):
    rips_complex=gd.RipsComplex(points=M,max_edge_length=12)
    Rips_simplex_tree=create_simplex_tree(M,dims)
    diag_Rips=Rips_simplex_tree.persistence()
    return diag_Rips


def compute_Bottleneck_distance(M,N,maxdim,dims):
    C=plot_persistence_diagram(M,maxdim)
    D=plot_persistence_diagram(N,maxdim)
    
    C1=[]
    for x,y in C:
        if sum(dims==x)>0:
            C1.append(y)
    D1=[]
    for x,y in D:
        if sum(dims==x)>0:
            D1.append(y)
    bottleneck_distance=gd.bottleneck_distance(C1,D1)
    return bottleneck_distance


def make_dists(x2):
    N = len(x2)
    Dists = zeros((N,N))
    for i in range(N):
        Dists[i,:] = abs(x2[i] - x2)
    return Dists

def dist2PD(distance_matrix,max_dim=2,max_edge_length=1):

    rips_complex = gd.RipsComplex(distance_matrix = distance_matrix,max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)        
    diag = simplex_tree.persistence()
    diag = [diag[i] for i in where([s[0]<max_dim for s in diag])[0]]
    return simplex_tree,diag



def rowvect2D(rowvec):
    points=[]
    X = rowvec
    Y = np.zeros((2,len(X)))
    Y[0,:] = X
    Y[1,:] = np.zeros((1,len(X))) 
    for i in range(len(X)):
        points.append(Y.T[i,:])
    points=np.array(points)
    return points
       
def min_matrix(M,m,n):
    K=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            K[i][j]=min(M[i][j],M[j][i])
    return (K)


def max_matrix(M,m,n):
    K=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            K[i][j]=max(M[i][j],M[j][i])
    return (K)


def make_k_community(k,i,points):
    a=[]
    for j in range(len(points)):
        if rank_matrix(points)[i,j]<=k:
            a.append(j)
    a=np.array(a) 
    return a


def get_local_filtration(points,filtration_matrix,filtration_steps,m,nodes):
    G = nx.Graph()
    positions = {}
    for i, pt in enumerate(points):
        positions[i] = pt
    #positions.append(pt)
    H=make_k_community(m,nodes,points)
    G_ex = nx.Graph()
    G_ex.add_nodes_from(range(len(points)))
    G_ex.add_edges_from(itertools.combinations(H, 2))
    filtration=[]
    for k,step in enumerate(filtration_steps):
        A = np.array( filtration_matrix <= step,int)
        for i in range(len(A)):
            A[i,i] = 0
        G = nx.to_networkx_graph(A)
        R = nx.intersection(G,G_ex)
        #H = nx.ego_graph(G,nodes,radius=m)
        filtration.append(R)
    return filtration


def bottleneck(diag1,diag2):
    diag1_simplicies=[[],[]]
    diag2_simplicies=[[],[]]
    for simplex in diag1: #Separates the first simplicial complex into 0 and 1 simplicies
        diag1_simplicies[simplex[0]].append(simplex[1]) 
    for simplex in diag2: #Separates the second simplicial complex into 0 and 1 simplicies
        diag2_simplicies[simplex[0]].append(simplex[1])
    b0=gd.bottleneck_distance(diag1_simplicies[0],diag2_simplicies[0]) #Calculates the bottleneck distance for 0
                                                                       #simplicies
    b1=gd.bottleneck_distance(diag1_simplicies[1],diag2_simplicies[1]) #Calculates the bottleneck distance for 1 
                                                                       #simplicies
    return array([b0,b1]).T


def matrix_local(i,k,points):
    A=rank_matrix(points)
    n=len(A)
    B= np.zeros((n,n))
    for j in range(len(A)):
        if j<=k:
            B[i][j]=A[i][j]
        else: 
            B[i][j]=0
    C=max_matrix(B,n,n)
    return (C)       


def run_pagerank_save_iterations(x0,T,Al,N):
    x=x0.copy()
    previous_r = x
    X = np.zeros((T,N))
    for i in range(0,T):
        X[i,:] = x
        x = x*Al
        x = x/sum(x)
        
    return X


def plot_ranks_vs_time(X):

    ranks = zeros(shape(X))
    for t in range(len(X)):
        ranks[t] = argsort(-X[t])

    f1 = figure(figsize = (25,3))
    plt.imshow(ranks.T+1,cmap='Oranges');
    caxis = colorbar()
    plt.title('Node Rank vs time')
    plt.ylabel('Node ID')
    plt.xlabel('time, t')
    
    return ranks


def node_smallest_neighbor(X,t,k):    
    # b = min(enumerate(X[t]),key=itemgetter(1))[0]
    # a = heapq.nsmallest(k,enumerate(X[t]),key=operator.itemgetter(1))
    # x = [x[0] for x in a]
    l = np.argsort( X[t] )
    x = l[:k]
    
    return x

    
def node_largest_neighbor(X,t,k):
    l = np.argsort( X[t] )
    x = l[-k:]
    
    return x


def get_k_neighbors(X,t,k,i):
    l = np.argsort( abs(X[t] - X[t][i]) )
    x = l[-k:]
    
    return x


