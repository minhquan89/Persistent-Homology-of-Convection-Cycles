# Multiplex Markov Chains

# Dane Taylor, Februaru, 2020


from pylab import *

import networkx as nx
import numpy as np
from scipy.linalg import block_diag
from scipy import sparse



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



 ### Example Network Models

def star(N):# star graph in which node 1 is the hub
	A = np.zeros((N,N))
	A[0,1:] = np.ones(N-1)
	A += A.T
	return A

def undirected_chain(N):#chain is k-regular by adding self edges at endpoints
	A = np.zeros((N,N))
	for i in range(1,N):
	    A[i-1,i] = 1
	A += A.T
	A[0,0] = 1
	A[-1,-1] = 1    
	return A

def chain_Markov(I,a=0):
    if I==1:
        Pt = 1
    if I==2:
        Pt = array([[1-a,a],[a,1-a]])
    if I>2:
        Pt = diag((1-a)**(ones(I)))
        Pt += diag((a)/2**(ones(I-1)),-1)
        Pt += diag((a)/2**(ones(I-1)),1)
        Pt[0,0] += (a)/2
        Pt[-1,-1] += (a)/2
    
    return Pt

def get_chain_Pts(I,a,N):
    if type(a)==float or type(a)==int:
        Pts = [chain_Markov(I,a) for n in range(N)]
    if type(a) == np.ndarray:
        Pts = [chain_Markov(I,a[n]) for n in range(N)]
    return Pts

def get_random_Pts(I,a,N):
    Pts = []
    for n in range(N):
        aa = rand()*(1-a) + a
        #Pt = np.array([[1-aa,aa],[aa,1-aa]],dtype=float)
        Pt = chain_Markov(I,aa)
        Pts.append( Pt )
    return Pts

def get_increasing_Pts(I,a,N):
    Pts = []
    for n in range(N):
        aa = n*(1-a)/(N-1) + a
        #Pt = np.array([[1-aa,aa],[aa,1-aa]],dtype=float)
        Pt = chain_Markov(I,aa)
        Pts.append( Pt )
    return Pts

def get_decreasing_Pts(I,a,N):   
    Pts = get_increasing_Pts(I,a,N)
    Pts.reverse()
    return Pts


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

def get_edge_colors(A,matrix):
        C = (np.array(np.where(A)))[0]
        D = (np.array(np.where(A)))[1]
        m = np.zeros(len(C))
        for t in range(len(C)):
            m[t] = matrix[C[t],D[t]]
    
        return m
    
def get_filtration_matrix(XX,max_value=.1,numsteps=8):
    
    #symmetrize input matrix
    m,n = shape(XX)
    XX = max_matrix(XX,m,n)
    
    #filtration_matrix = (XX==0)*np.max(XX)*10 + XX #make non-edges some huge number
    filtration_matrix = (XX==0)*max_value + XX #make non-edges some huge number
    filtration_matrix = filtration_matrix  - np.diag(np.diag(filtration_matrix))
    return filtration_matrix, np.linspace(0,np.max(XX),numsteps)



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


def get_filtration_matrix_3(XX,max_value=.1,numsteps=8):
    
    #symmetrize input matrix
    m,n = shape(XX)
    XX = max_matrix(XX,m,n)
    
    fil_matrix = max_value - XX
    fil_matrix = fil_matrix - np.diag(np.diag(fil_matrix))
    
#    return fil_matrix, max_value - np.linspace(0,max_value,numsteps)
    return fil_matrix, max_value - np.linspace(.003,max_value*.95,numsteps)

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


### Supracentrality

def build_block_diag(As):# returns diag(A^{(t)})
    As_block = block_diag(As[0],As[1]) 
    for n in range(2,len(As)):
        As_block = block_diag(As_block,As[n]) 
    return As_block

def build_sum_term(Pts,N):    
    if len(Pts) == N:
        T = len(Pts[0])
        X = zeros((N*T,N*T))
        for n in range(N):
            En = np.diag(np.arange(N)==n)*1
            X += kron(Pts[n],En)

        return X

    if len(Pts) == 1: return kron(Pts[0],eye(N))
    
    if len(Pts) != N:
        print('error! Wrong number of interlayer Markov chains.')
        return 0
    
    return 0

def supraCentralityMatrix(As,Pts,w,alpha):
    G = []
    for A in As:
          G.append(google_matrix(A,alpha)) #Determines Google matrices
    C = (1-w)*build_block_diag(G) + w*build_sum_term(Pts,len(As[0])) #Determines Supracentrality matirx

    return C

def supraCentrality(As,Pts,w,alpha):
    C = supraCentralityMatrix(As,Pts,w,alpha)
    joint = dom_left_eig(C)
    
    return joint.T.reshape(len(Pts[0]),len(As[0])).T #Reshapes eigenvalue




### Perturbation theory for small and large omega

def predicted_strong(As,Pts,alpha):
    
    #compute dominant eigenvectors of interlayer Markov chains 
    vs = np.zeros( (len(Pts),len(As)) )
    for n in range(len(Pts)):
        v = dom_left_eig(Pts[n])
        v /= sum(v)
        vs[n,:] = v
        
    #obtain an effective intralayer Markov chain
    X = zeros(shape(As[0]))
    for i in range(len(As)):
        X += np.dot( diag(vs[:,i]),google_matrix(As[i],alpha) )

    #use dominant eigenvector to get the weights
    weights = dom_left_eig(X)
    weights /= sum(weights)

    #use dominant eigenvector to obtain stationary distribution
    x2_strong = np.zeros(( len(Pts),len(As) ))
    for n in range(len(Pts)):
        x2_strong[n,:] = weights[n]* vs[n,:]

    return x2_strong,weights

def predicted_weak(As,Pts,alpha):
    
    #compute dominant eigenvectors of intralayer Markov chains 
    vs = np.zeros(( len(Pts),len(As) ))
    for i in range( len(As) ):
        v = dom_left_eig(google_matrix(As[i],alpha) )
        v /= sum(v)
        vs[:,i] = v
        
    #obtain an effective interlayer Markov chains 
    X_tilde = zeros( shape(Pts[0]) )
    for n in range(len(Pts)):
        X_tilde += np.dot(diag(vs[n,:]), Pts[n] )

    #use dominant eigenvector to get the weights
    weights = dom_left_eig(X_tilde)
    weights /= sum(weights)

    #use weights and dominant eigenvectors to obtain stationary distribution
    x2_weak = np.zeros(( len(Pts),len(As) ))
    for i in range(len(As)):
        x2_weak[:,i] = weights[i]* vs[:,i]

    return x2_weak,weights





### Study imbalance, convection and optimality

def study_imbalance(As,Pts,w,alpha):
    x2  = supraCentrality(As,Pts,w,alpha)    
    F =  np.dot(np.diag(x2.T.flatten()),supraCentralityMatrix(As,Pts,w,alpha))

    Delta = (F - F.T)
    total_imbalance = norm(Delta,'fro')

    pos_DF = np.maximum(F-F.T, np.zeros(np.shape(F)))

    return F,pos_DF,Delta,total_imbalance


def get_optimal_curves(ws,a_s,funs,As,alpha):
    N = len(As[0])
    I = len(As)

    imb_opts = zeros(( len(funs),len(a_s) ))
    conv_opts = zeros(( len(funs),len(a_s) ))
    conv_rates = np.zeros(( len(funs),len(a_s),len(ws) ))
    total_imbalances = np.zeros(( len(funs),len(a_s),len(ws) ))
    
    for tt,fun in enumerate(funs):
        for t,a in enumerate(a_s):
            Pts = fun(I,a,N) 
            
            for i,w in enumerate(ws):
                P = supraCentralityMatrix(As,Pts,w,alpha)
                conv_rates[tt,t,i] = - sort(-real(linalg.eig(P)[0]))[1]        
                _,_,_,total_imbalances[tt,t,i] = study_imbalance(As,Pts,w,alpha)
            
            imb_opts[tt,t] = ws[argmax(total_imbalances[tt][t])]
            conv_opts[tt,t] = ws[argmin(conv_rates[tt][t])]
        
    return conv_rates,total_imbalances,imb_opts,conv_opts


def compare_d_Delta(As,Pts,ws,alpha):
    I = len(As)
    N = len(Pts)
    dd =  array([sum(A,1) for A in As]).reshape(N*I)
        
    boo1 = []
    boo2 = []    
    tots = zeros(len(ws))
    Pearsons = zeros((len(ws),2))
    for t,w in enumerate(ws):
        #print(w)

        _,_,Delta,tots[t] = study_imbalance(As,Pts,w,alpha)
        
        AA = np.triu(build_block_diag(As))#intralayer edges
        row,col,weights = sparse.find(sparse.coo_matrix(AA!=0))
        delta = array([Delta[row[p],col[p]] for p in range(len(row))])
        d = array([ dd[row[p]] - dd[col[p]] for p in range(len(row))])
                
        AA = np.triu(build_sum_term(Pts,N))#interlayer edges
        row,col,weights = sparse.find(sparse.coo_matrix(AA!=0))
        delta2 = array([Delta[row[p],col[p]] for p in range(len(row))])
        d2 = array([ dd[row[p]] - dd[col[p]] for p in range(len(row))])
        
        Pearsons[t,0] = np.corrcoef(d,delta)[0,1]        
        Pearsons[t,1] = np.corrcoef(d2,delta2)[0,1]
        boo1.append(delta)
        boo2.append(delta2)
        
        print('Pearsons = '+ str(Pearsons[t,0]) + ', ' + str(Pearsons[t,1]))
        
    return Pearsons,tots,boo1,boo2

