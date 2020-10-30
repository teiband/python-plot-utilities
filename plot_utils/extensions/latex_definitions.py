import matplotlib as mpl

mpl.use('WebAgg')  # web backend for plotting, works only locally but not via VNC

from matplotlib import rc, rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rc('text', usetex=True)
