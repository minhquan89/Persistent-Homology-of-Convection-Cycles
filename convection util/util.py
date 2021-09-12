# EVC filtrations for flows over networks

# Minh Quang Le, 9/10/2021


from pylab import *

import networkx as nx
import numpy as np
from scipy.linalg import block_diag
from scipy import sparse
import pandas as pd


### Basic Network Concepts

def transition_matrix(A): #creates a transition matrix from an adjacency matrix
	N = len(A)
	Dinv = zeros((N,N))
	for i in range(N): 
	    if sum(A[i])>0:
	        Dinv[i,i] = 1/sum(A[i])
	P = dot(Dinv,A)
	return P    

def google_matrix(A,alpha): #creates the PageRank, or Google, matrix
	P = transition_matrix(A)
	G = alpha*P + (1-alpha)/len(A) * ones(shape(A))    
	return G

def dom_left_eig(GG): #implements power method to compute dominant left eigenvector
	eigenValues,eigenVectors = eig(GG.T)        
	idx = eigenValues.argsort()[-1]   
	eigenValues = eigenValues[idx]
	x = np.abs(eigenVectors[:,idx])
	return x/sum(x)

def pagerank(A,alpha):  #computes pagerank centrality
	G = google_matrix(A,alpha)
	return dom_left_eig(G)



def make_toy_graph():
    G = nx.DiGraph()
    # nodes and their coordinates
    G.add_node('A',pos=(0.50607613,-0.86811485))
    G.add_node('B',pos=(0.58601273,0.0335377 ))
    G.add_node('C',pos=(0.1276708 ,0.56052309))
    G.add_node('D',pos=(-0.03736543,-0.31509468))
    G.add_node('E',pos=(-0.46760338,-0.97321343))
    G.add_node('F',pos=(-0.36076138,0.56236217))
    G.add_node('G',pos=(0.04597052,-1.))

    # weighted edges
    G.add_edges_from([('A', 'B')], weight=4)
    G.add_edges_from([('A','G')], weight=1)
    G.add_edges_from([('B','C')], weight=3)    
    G.add_edges_from([('B','D')], weight=1)
    G.add_edges_from([('C','B')], weight=3)
    G.add_edges_from([('C','D')], weight=3)
    G.add_edges_from([('D','A')], weight=4)
    G.add_edges_from([('D','B')], weight=1)
    G.add_edges_from([('D','C')], weight=2)
    G.add_edges_from([('D','E')], weight=2)
    G.add_edges_from([('E','F')], weight=2)
    G.add_edges_from([('E','G')], weight=1)
    G.add_edges_from([('F','C')], weight=2)    
    G.add_edges_from([('G','A')], weight=1)

    
    return G
    


def draw_toy_graph(G,vmin,vmax,node_size,node_colors,edge_colors,edge_width,node_cmap,edge_cmap,s,ax,color_title=''):
   
    pos = nx.get_node_attributes(G,'pos')
    #vmin=0#min(edge_colors)
    #vmax=1#max(edge_colors)

    
    nodes_draw = nx.draw_networkx_nodes(G,
                       pos=pos, 
                       node_color = node_colors, 
                       node_size=node_size,
                       cmap=node_cmap,
                       vmin=vmin,
                       alpha=1,
                       vmax=vmax,
                       label=list(G.nodes()),
                       ax=ax)
    nodes_draw.set_edgecolor('k')

    edges_draw = nx.draw_networkx_edges(G,
                       pos=pos, 
                       edge_color=edge_colors,
                       edge_cmap=edge_cmap,
                       edge_vmin=vmin,
                       edge_vmax=vmax,
                       node_size=node_size,
                       ax=ax,
                       width=edge_width,
                       alpha=1,
                       arrows=True,
                       arrowsize=20)

    edge_labels = dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])
    edgelabels_draw = nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
    nodeslabels_draw = nx.draw_networkx_labels(G,pos)
    #fig1.tight_layout(nodes_draw)
    ax.axis('off')
    #ax.set_xlim([-1,1])
    #ax.set_ylim([-2,1])
    # make colorbar
    sm = plt.cm.ScalarMappable(cmap=s, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    sm._A = []
    plt.colorbar(sm,label=color_title)
    
    return 

def get_pageRank_net_flow(A,alpha):
    p_rank = pagerank (A,alpha)
    P = google_matrix (A,alpha)
    F =  np.dot(np.diag(p_rank),P) # net flows over edges
    return F

def get_pageRank_net_flow_2(A,alpha):
    p_rank,P = pagerank1 (A,alpha)
    #P = google_matrix (A,alpha)
    F =  np.dot(np.diag(p_rank),P) # net flows over edges
    return F


def get_edge_colors(A,matrix):
        C = (np.array(np.where(A)))[0]
        D = (np.array(np.where(A)))[1]
        m = np.zeros(len(C))
        for t in range(len(C)):
            m[t] = matrix[C[t],D[t]]
    
        return m

 
def make_wfiltration_fig(G,nets,titles,y_label):
    
    pos = nx.get_node_attributes(G,'pos')
    
    fig, ax = plt.subplots(1,len(nets),figsize=(16,3))
#    ax = axes.flatten()

    for i in range(len(nets)):
        nx.draw_networkx(nets[i], 
                         pos=pos,
                         ax=ax[i], 
                         node_size=800,
                         #font_size=19,
                         node_color='lightgray')  
        ax[i].set_title(titles[i])
        ax[i].set_xlim([-1,1])
        ax[i].set_ylim([-1.5,1])        
        ax[i].axis('off')
        if i==0:
             ax[i].set_ylabel(y_label)
#    plt.show()
    plt.tight_layout();
    #savefig('VR_filtration.pdf');
    return fig,ax

def draw_toy_graph_1(G,vmin,vmax,node_size,node_colors,edge_colors,edge_width,node_cmap,edge_cmap,s,ax,color_title,pos):
   
   # pos = dict( (n, n) for n in G.nodes() )  
    nodes_draw = nx.draw_networkx_nodes(G,
                       pos=pos, 
                       node_color = node_colors, 
                       node_size=node_size,
                       cmap=node_cmap,
                       vmin=vmin,
                       alpha=1,
                       vmax=vmax,
                       label=list(G.nodes()),
                       ax=ax)
    nodes_draw.set_edgecolor('k')

    edges_draw = nx.draw_networkx_edges(G,
                       pos=pos, 
                       edge_color=edge_colors,
                       edge_cmap=edge_cmap,
                       edge_vmin=vmin,
                       edge_vmax=vmax,
                       node_size=node_size,
                       ax=ax,
                       width=edge_width,
                       alpha=1,
                       arrows=True,
                       arrowsize=20,
                       connectionstyle='arc3, rad = .2')

    ax.axis('off')
    return 






def draw_toy_undirected_graph(G,vmin,vmax,node_size,node_colors,edge_colors,edge_width,node_cmap,edge_cmap,s,ax):
   
    pos = nx.get_node_attributes(G,'pos')
    #vmin=0#min(edge_colors)
    #vmax=1#max(edge_colors)

    
    nodes_draw = nx.draw_networkx_nodes(G,
                       pos=pos, 
                       node_color = node_colors, 
                       node_size=node_size,
                       cmap=node_cmap,
                       vmin=vmin,
                       alpha=1,
                       vmax=vmax,
                       label=list(G.nodes()),
                       ax=ax)
    nodes_draw.set_edgecolor('k')

    edges_draw = nx.draw_networkx_edges(G,
                       pos=pos, 
                       edge_color=edge_colors,
                       edge_cmap=edge_cmap,
                       edge_vmin=vmin,
                       edge_vmax=vmax,
                       node_size=node_size,
                       ax=ax,
                       width=edge_width,
                       alpha=1,
                       arrows=False,
                       arrowsize=20)

    edge_labels = dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])
    edgelabels_draw = nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
    nodeslabels_draw = nx.draw_networkx_labels(G,pos)
    ax.axis('off')
    ax.set_xlim([-1,1])
    ax.set_ylim([-2,1])    
    #fig1.tight_layout(nodes_draw)
    
    # make colorbar
#     sm = plt.cm.ScalarMappable(cmap=s, norm=plt.Normalize(vmin = vmin, vmax=vmax))
#     sm._A = []
#     plt.colorbar(sm)
    
    return 






def draw_toy_graph2(G,vmin,vmax,node_size,node_colors,edge_colors,edge_width,node_cmap,edge_cmap,s,ax,color_title=''):
   
    pos = nx.get_node_attributes(G,'pos')
    #vmin=0#min(edge_colors)
    #vmax=1#max(edge_colors)

    
    nodes_draw = nx.draw_networkx_nodes(G,
                       pos=pos, 
                       node_color = node_colors, 
                       node_size=node_size,
                       cmap=node_cmap,
                       vmin=vmin,
                       alpha=1,
                       vmax=vmax,
                       label=list(G.nodes()),
                       ax=ax)
    nodes_draw.set_edgecolor('k')

    edges_draw = nx.draw_networkx_edges(G,
                       pos=pos, 
                       edge_color=edge_colors,
                       edge_cmap=edge_cmap,
                       edge_vmin=vmin,
                       edge_vmax=vmax,
                       node_size=node_size,
                       ax=ax,
                       width=edge_width,
                       alpha=1,
                       arrows=True,
                       arrowsize=20)

    edge_labels = dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])
    #edgelabels_draw = nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,ax=ax)
    nodeslabels_draw = nx.draw_networkx_labels(G,pos,ax=ax)
    #fig1.tight_layout(nodes_draw)
    ax.axis('off')
    #ax.set_xlim([-1,1])
    #ax.set_ylim([-2,1])
    # make colorbar
    sm = plt.cm.ScalarMappable(cmap=s, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    #sm._A = []
    #plt.colorbar(sm,label=color_title)
    
    return 


def convert_F_to_small_F(F):

    small_F = np.zeros((int(len(F)/4),int(len(F)/4)))

    #node 0 (left square, top row)
    small_F[0,1] = F[6,5]
    small_F[1,0] = F[13,14]
    small_F[0,4] = F[14,22]
    small_F[4,0] = F[23,15]

    #node 1 (left square, second row)
    small_F[1,2] = F[4,3]
    small_F[2,1] = F[11,12]
    small_F[1,5] = F[12,20]
    small_F[5,1] = F[21,13]

    #node 2 (left square, third row)
    small_F[2,3] = F[2,1]
    small_F[3,2] = F[9,10]
    small_F[2,6] = F[10,18]
    small_F[6,2] = F[19,11]

    #node 3 (left square, fourth row)
    #small_F[3,4] = F[2,1]
    #small_F[4,3] = F[9,10]
    small_F[3,7] = F[8,16]
    small_F[7,3] = F[17,9]

    #node 4 (left-middle square, first row)
    small_F[4,5] = F[22,21]
    small_F[5,4] = F[29,30]
    small_F[4,8] = F[30,38]
    small_F[8,4] = F[39,31]

    #node 5 (left-middle square, second row)
    small_F[5,6] = F[20,19]
    small_F[6,5] = F[27,28]
    small_F[5,9] = F[28,36]
    small_F[9,5] = F[37,29]

    #node 6 (left-middle square, third row)
    small_F[6,7] = F[18,17]
    small_F[7,6] = F[25,26]
    small_F[6,10] = F[26,34]
    small_F[10,6] = F[35,27]


    #node 7 (left-middle square, third row)
    #small_F[7,8] = F[2,1]
    #small_F[8,7] = F[9,10]
    small_F[7,11] = F[24,32]
    small_F[11,7] = F[33,25]


    #node 8 (left-middle square, third row)
    small_F[8,9] = F[38,37]
    small_F[9,8] = F[45,46]
    small_F[8,12] = F[46,54]
    small_F[12,8] = F[55,47]

    #node 9 (left-middle square, third row)
    small_F[9,10] = F[36,35]
    small_F[10,9] = F[43,44]
    small_F[9,13] = F[44,52]
    small_F[13,9] = F[53,45]

    #node 10 (left-middle square, third row)
    small_F[10,11] = F[34,33]
    small_F[11,10] = F[41,42]
    small_F[10,14] = F[42,50]
    small_F[14,10] = F[51,43]

    #node 11 (left-middle square, third row)
    ##small_F[11,12] = F[2,1]
    #small_F[12,11] = F[9,10]
    small_F[11,15] = F[40,48]
    small_F[15,11] = F[49,41]



    #node 12 (left-middle square, third row)
    small_F[12,13] = F[54,53]
    small_F[13,12] = F[61,62]

    #node 13 (left-middle square, third row)
    small_F[13,14] = F[52,51]
    small_F[14,13] = F[59,60]

    #node 14 (left-middle square, third row)
    small_F[14,15] = F[50,49]
    small_F[15,14] = F[57,58]

    #node 15 (left-middle square, third row)

    return small_F

def pagerank2 (A,alpha):
    #P = google_matrix (A,alpha)
    P = A.copy()
    for i in range(len(A)):
        d_i = sum(A[i,:])
        print(d_i)
        P[i,i] = 1 - d_i
        print(P[i,i])
    x = np.zeros(len(A))
    x[0] = 1
    x[1] = 1
    x[2] = 1
    x[3] = 1
    x[4] = 1
    x[5] = 1
    x[6] = 1
    x[7] = 1
    x[8] = 1
    x[15] = 1
    x[16] = 1
    x[23] = 1
    x[24] = 1
    x[31] = 1
    x[32] = 1
    x[39] = 1
    x[40] = 1
    x[47] = 1
    x[48] = 1
    x[55] = 1
    x[56] = 1
    x[57] = 1
    x[58] = 1
    x[59] = 1
    x[60] = 1
    x[61] = 1    
    x[62] = 1
    x[63] = 1
    
    x = x/sum(x)
    #print(sum(x))
    for t in range(100000):
        x = np.dot(P.T,x)
        #print(sum(x))
    return x,P  

def pagerank1 (A,alpha):
#    P = google_matrix (A,alpha)
    P = A.copy()
    for i in range(len(A)):
        d_i = sum(A[i,:])
  #      print(d_i)
        P[i,i] = 1 - d_i
 #       print(P[i,i])

    x = np.ones(len(A))
    x = x/sum(x)
    for t in range(10000):
         x = np.dot(P.T,x)
    return x,P 


