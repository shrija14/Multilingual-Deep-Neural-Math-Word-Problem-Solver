def beautifyPlot():
    import matplotlib.pyplot as pp
    pp.clf()
    pp.rcParams['font.family'] = 'sans-serif'
    #   pp.rcParams['font.sans-serif'] = 'Roboto'
    pp.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
    pp.rcParams['font.family'] = 'sans-serif'
#     pp.rcParams['text.usetex'] = False
#     pp.rcParams['text.latex.unicode']=False
    pp.rcParams['font.sans-serif'] = 'cm'
    pp.rcParams['font.size'] = 36
    pp.rcParams['text.color'] = "#000000"
    pp.rcParams['ytick.labelsize'] = 32
    pp.rcParams['xtick.labelsize'] = 32
    pp.rcParams['ytick.color'] = '#000000'
    pp.rcParams['xtick.color'] = '#000000'
    pp.rcParams['legend.fontsize'] = 32
    pp.rcParams['lines.markersize'] = 14
    pp.rcParams['axes.titlesize'] = 40
    pp.rcParams['axes.labelcolor'] = '#000000'
    pp.rcParams['axes.labelsize'] = 36
    #     plt.rcParams['axes.edgecolor'] = '#f0f0f0'
    pp.rcParams['axes.edgecolor'] = '#525252'
    pp.rcParams['axes.linewidth'] = 1.0
    pp.rcParams['axes.grid'] = False
    #     plt.rcParams['axes.grid'] = True
    #     plt.rcParams['axes.grid.axis'] = "y"
    #     plt.rcParams['grid.linewidth'] = 3.0
    pp.rcParams['grid.color'] = "#FFFFFF"
    pp.rcParams['legend.frameon'] = True
    pp.rcParams['legend.framealpha'] = 0.1
    pp.rcParams['legend.fancybox'] = True    
    pp.rcParams['legend.numpoints'] = 1
    pp.rcParams['legend.scatterpoints'] = 1
    pp.rcParams['legend.facecolor'] = 'none'
    pp.rcParams['figure.figsize'] = 8,8
    pp.gca().spines['top'].set_visible(False)
    pp.gca().spines['bottom'].set_visible(True)
    pp.gca().spines['right'].set_visible(False)
    pp.gca().spines['left'].set_visible(True)
    pp.gca().get_xaxis().tick_bottom()
    pp.gca().get_yaxis().tick_left()
    pp.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=True, left=False,
                    right=False, labelleft=True, length=10, width=2, direction='out',  color='#ffffff')
    #   pp.tick_params(axis='both', which='major', bottom=True, top=False, labelbottom=True, left=True,
    #                   right=False, labelleft=True, length=10, width=2, direction='out',  color='#636363')
    return pp