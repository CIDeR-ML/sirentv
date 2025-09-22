import numpy as np
import matplotlib.pyplot as plt

try:
    from fast_histogram import histogram1d
except ImportError:
    print('fast_histogram not installed, using numpy')
    histogram1d = lambda *args,**kwargs: np.histogram(*args,**kwargs)[0]
    
    
def hist(a, bins=None, **kwargs):
    log = False
    if bins[1]-bins[0] != bins[2]-bins[1]:
        log = True
        
    data = np.log10(a+a[a!=0].min()/10) if log else a
        
    range = [bins.min(),bins.max()]
    if log:
        range = list(map(np.log10, range))
    bins  = len(bins)
    
    counts = histogram1d(data, bins=bins, range=range).astype(float)
    edges = np.linspace(*range, bins+1)
    
    if kwargs.pop('density', False):
        counts /= np.diff(edges) * counts.sum()

    if log:
        edges = 10**edges
    centers = (edges[1:]+edges[:-1])/2
        
    plt.gca().plot(centers, counts, drawstyle='steps-mid', **kwargs)
        
    