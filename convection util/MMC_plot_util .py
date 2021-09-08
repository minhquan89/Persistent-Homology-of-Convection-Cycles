# Multiplex Markov Chains

# Dane Taylor, Februaru, 2020


from pylab import *
import networkx as nx
from scipy.linalg import block_diag
from scipy import sparse

from MMC_util import *

### Plotting

def norm_cmap(x,vmin,vmax):
    return (log(x) - log(vmin) ) / ( log(vmax) -  log(vmin) )



def plot_stationary_curves(x,As,Pts,ws,alpha):
    f4, ax = plt.subplots(len(ws),2,figsize=(6,6),sharex='col',sharey='col')
    for t,w in enumerate(ws):
        x = supraCentrality(As,Pts,w,alpha)
        ax[t,0].plot(x.T,alpha=.5);
        ax[t,1].plot(x,alpha=.5);
        ax[t,0].set_ylabel('$\omega='+str(w) + '$\n\n $\pi_n^{(i)}$')
    ax[-1,0].set_xlabel('layer, $i$')
    ax[-1,1].set_xlabel('node, $n$')
    ax[0,0].set_title('Stationary Distribution')
    ax[0,1].set_title('Stationary Distribution')
    plt.tight_layout() 
    return f4, ax


def plot_rank(As,x2 = 0.01*np.ones((11,2)),vmin=.025,vmax=.25):
	N = len(As[0])
	cmap = plt.cm.get_cmap('hot')    
	   
	f1 = plt.figure(figsize = (4,8))
	pos = {}
	pos[0]  = np.array([0,0])
	for i in range(1,N):
	    theta = (2*pi*(i-1) ) / (N-1)
	    pos[i] = np.array([-np.cos(theta),np.sin(theta)])

	plt.subplot(2,1,1)
	G = nx.from_numpy_matrix(As[0])
	edges = nx.draw_networkx_edges(G, 
								   pos, 
								   width=9.0, 
								   edge_color='k', 
								   style='solid', 
								   alpha=.9 
								  )
	nodes = nx.draw_networkx_nodes(G, 
								   pos, 
								   linewidths=1,
								   node_size=900, 
								   node_color=cmap(norm_cmap(x2[:,0],vmin,vmax) ), 
								   node_shape='o', 
								   alpha=1, 
								   cmap=None
								  )

	limits = plt.axis('off')

	plt.subplot(2,1,2)
	G = nx.from_numpy_matrix(As[1])
	edges = nx.draw_networkx_edges(G, 
								   pos, 
								   width=9.0, 
								   edge_color='k', 
								   style='solid', 
								   alpha=0.9 
								  )
	nodes = nx.draw_networkx_nodes(G, 
								   pos, 
								   node_size=900, 
								   node_color=cmap(norm_cmap(x2[:,1],vmin,vmax) ), 
                                   node_shape='o', 
								   alpha=1, 
								   cmap=None
								  )
	limits = plt.axis('off')

	return


def plot_strong_compare(x2,x2_strong,w,vmin,vmax):
    x2 = reshape(x2,(1,prod(shape(x2))))[0]
    x2_strong = reshape(x2_strong,(1,prod(shape(x2))))[0]
    
    cmap = matplotlib.cm.get_cmap('hot')    

    ax = plt.gca()
    plt.scatter(x2_strong,x2,marker='<',s=70,c=cmap(norm_cmap(x2,vmin,vmax) ),edgecolors='k')
    lower = .01 #.4*np.min(x2)
    upper = 1 #.1*np.max(x2)
    plt.plot([lower,upper],[lower,upper],'k--',linewidth=1)
    ax.set_yscale('log');ax.set_xscale('log')
    ax.set_xlim([lower,upper]);ax.set_ylim([lower,upper]);
    ax.set_title('Aggregation, $\omega\mapsto 1$');
    ax.set_xlabel(r'predicted $\pi_n^{(i)}(\omega)$')
    ax.set_ylabel(r'observed $\pi_n^{(i)}(\omega)$');
    return


def plot_weak_compare(x2,x2_weak,w,vmin,vmax):
    x2 = reshape(x2,(1,prod(shape(x2))))[0]
    x2_weak = reshape(x2_weak,(1,prod(shape(x2))))[0]
    
    cmap = matplotlib.cm.get_cmap('hot')    

    ax = plt.gca()
    a = plt.scatter(x2_weak,x2,marker='s',s=100,c=cmap(norm_cmap(x2,vmin,vmax) ),edgecolors='k')
    lower = .01 
    upper = 1 
    plt.plot([lower,upper],[lower,upper],'k--',linewidth=1)
    ax.set_yscale('log');ax.set_xscale('log')
    ax.set_xlim([lower,upper]);ax.set_ylim([lower,upper]);
    ax.set_title('Decoupling, $\omega \mapsto 0$');
    ax.set_xlabel(r'predicted $\pi_n^{(i)}(\omega)$')
    ax.set_ylabel(r'observed $\pi_n^{(i)}(\omega)$');
    return


def make_asym_fig(fig_name,
	              ws,
	              As,
	              Pts,
	              alpha,
	              vmin,
	              vmax):
    
    f = figure(figsize=(6,3))
    
    x2 = supraCentrality(As,Pts,ws[0],alpha)
    x2_weak,weights_weak = predicted_weak(As,Pts,alpha)
    plt.subplot(1,2,1)
    plot_weak_compare(x2,x2_weak,ws[1],vmin,vmax)
    
    x22 = supraCentrality(As,Pts,ws[1],alpha)
    x2_strong,weights_strong = predicted_strong(As,Pts,alpha)
    plt.subplot(1,2,2)
    plot_strong_compare(x22,x2_strong,ws[1],vmin,vmax)

    plt.tight_layout()    
    text(0.0000014, 1.2, '(A)',size=14)
    text(0.0015, 1.2, '(B)',size=14)

    plt.savefig(fig_name)
    return  x2,x2_weak,x22,x2_weak




def plot_optimals(ws,a_s,funs,convs,tots,exp_names):

    f, (ax1,ax2) = plt.subplots(2,len(funs),figsize=(7,3),sharex='col',sharey='row')
    strr = ['-','--',':','-.','-']

    for tt in range(len(funs)):    
        for t in range(len(a_s)):
            
            ax2[tt].plot(ws,convs[tt][t])#,strr[t],linewidth='2')
            ax1[tt].plot(ws,tots[tt][t])#,strr[t],linewidth='2')
            #plt.xlim([0,1])
            #plt.ylim([0,.03])            

    ax1[0].set_ylabel('$||\Delta||_{F}$')
    ax2[0].set_ylabel('$\lambda_2(\omega)$')

    for t in range(len(funs)): 
        ax1[t].set_title(exp_names[t])
        ax2[t].set_xlabel('$\omega$')
        ax2[t].set_xlim([0,1])
        ax2[t].set_ylim([.7,1])     
        ax1[t].set_xlim([0,1])
        ax1[t].set_ylim([0,.035])    

    #ax1[0].legend(['$a='+str(a)+'$' for a in a_s],loc='lower center')

    #subplot(2,3,4)
    text(-4.25, 1.35, '(A)',size=14)
    text(-4.25, 0.95, '(B)',size=14)


    plt.tight_layout()
    plt.legend(['$a='+str(a)+'$' for a in a_s],loc='lower center',bbox_to_anchor=(-1.3, -0.9), shadow=True, ncol=len(a_s));
    
    
    return f,ax1,ax2


def plot_optimals_SM(a_s2,imb_opts,conv_opts,exp_names):
    f1 = figure(figsize=(8,2.9))

    yrange = [.4,.8]
    strr = ['o','s','<','x']
    subplot(1,3,1)
    for i in range(len(conv_opts)):plt.scatter(a_s2,imb_opts[i],marker=strr[i],s=20,alpha=0.75);
    plt.legend(exp_names)    
    plt.title('Imbalance')
    plt.xlabel('$a$')
    plt.ylim(yrange)
    plt.ylabel('$\omega^*_{\Delta}$')

    subplot(1,3,2)
    for i in range(len(conv_opts)):plt.scatter(a_s2,conv_opts[i],marker=strr[i],s=20,alpha=0.75);
    plt.title('Convergence')
    plt.xlabel('$a$')
    plt.ylim(yrange)
    plt.ylabel('$\omega^*_{\lambda_2}$')
    #plt.legend(exp_names)


    subplot(1,3,3)
    for i in range(len(conv_opts)):plt.scatter(imb_opts[i],conv_opts[i],marker=strr[i],s=20,alpha=0.75)
    #plt.legend(exp_names)
    
    plt.title('Comparison')
    plt.ylabel('$\omega^*_{\lambda_2}$')
    plt.xlabel('$\omega^*_{\Delta}$')
    plt.xlim(yrange)
    plt.ylim(yrange)
    plt.tight_layout()
    
    #plt.legend(exp_names,loc='lower center',bbox_to_anchor=(-1.26, -0.55), ncol=4);
    plt.plot([.3,1],[.3,1],':k') 
    
    
    
    text(-.94, .82, '(A)',size=14)
    text(-.36, .82, '(B)',size=14)
    text(.24, .82, '(C)',size=14)
    
    
    return


def plot_Delta_Curves(ws2,boo1,boo2):

    tots1 = zeros(len(boo1))
    tots2 = zeros(len(boo2))
    for e in range(len(boo1)):
        tots1[e] = norm(boo1[e],2)
        tots2[e] = norm(boo2[e],2)


    f4, ax = plt.subplots(1,3,figsize=(8,2.5),sharex='row')
    titles = ['Intralayer Edges','Interlayer Edges','Optimal Imbalance']      

    ll = min(5000,len(boo1[0])) # don't plot too many lines
    
    ax[0].plot(ws2,abs(array(boo1)[:,:ll]),alpha=.6);
    ax[1].plot(ws2,abs(array(boo2)[:,:ll]),alpha=.6);
    ax[0].set_ylabel('net flow, $\Delta_{pq}^+$')
    ax[1].set_ylabel('net flow, $\Delta_{pq}^+$')
    for t in range(3): 
        ax[t].set_xlabel('coupling, $\omega$')
        ax[t].set_title(titles[t])

    ax[2].plot(ws2,tots1,'k--')
    ax[2].plot(ws2,tots2,'gray')
    ax[2].set_ylabel('$||\Delta||_F$')
    ax[2].legend(['Intralayer','Interlayer'])
    
    plt.tight_layout()    

    return




