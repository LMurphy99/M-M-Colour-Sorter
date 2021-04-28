from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def makeGif(fig, ax, filename):
    def init():
        return fig,
    
    def animate(i):
        ax.view_init(10, 2*i)
        return fig,
        
    anim = animation.FuncAnimation(fig,animate,init_func=init,
                                  frames=180, interval=2, blit=True)
    anim.save(filename)
    
def cuboid_data2(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

def plotCubeAt2(positions,sizes=None,colors=None, **kwargs):
    if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( cuboid_data2(p, size=s) )
    return Poly3DCollection(np.concatenate(g),  
                            facecolors=np.repeat(colors,6), **kwargs)
                            
                            
def plot3DContour(mu,cov,factors,ax,**kwargs):
    evals, evecs = np.linalg.eig(cov)
    
    for factor in factors:
        radii = np.sqrt(evals)*factor

        # calculate cartesian coordinates for the ellipsoid surface
        u = np.linspace(0.0, 2.0 * np.pi, 60)
        v = np.linspace(0.0, np.pi, 60)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        for j in range(len(x)):
            for k in range(len(x)):
                [x[j,k],y[j,k],z[j,k]] = np.dot([x[j,k],y[j,k],z[j,k]], evecs.T) + mu

        ax.plot_surface(x, y, z,  rstride=3, cstride=3, alpha=0.15, shade=True, **kwargs)