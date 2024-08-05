import matplotlib as mpl

mpl.use('WebAgg')  # web backend for plotting, works only locally but not via VNC

from matplotlib import rc, rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['pdf.fonttype'] = 42 # allows only true type fonts
rcParams['ps.fonttype'] = 42 # allows only true type fonts
rc('text', usetex=True)
