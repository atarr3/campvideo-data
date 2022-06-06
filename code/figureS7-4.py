import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from matplotlib import rc

from os.path import join

# root folder for replication repo
ROOT = '..'

def main():
    # read in data
    res = pd.read_csv(join(ROOT, 'data', 'summary_results.csv'),
                      index_col='uid')
    # plot results
    plt.rcParams['font.sans-serif'] = 'Verdana'
    #rc('text', usetex=True)
    constrained_layout = True # use tight layout if False
    
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 4.333), sharey=False, 
                            constrained_layout=constrained_layout)
    
    for i in range(3):
        # plot scatter plot of n_true vs n_auto
        if i == 0:
            ticks = [0, 10, 20]
            # scatter plot and y=x line
            axs[i].scatter(res.n_auto, res.n_true, s=10, c='black')
            axs[i].plot([0, 25], [0, 25], 'k--', lw=1)
            
            # axis formatting
            axs[i].set_xlim(0, 25)
            axs[i].set_ylim(0, 25)
            axs[i].set_xlabel('Automated Summary', fontsize=12)
            axs[i].set_ylabel('Manual Summary', fontsize=12)
            axs[i].tick_params(labelsize=10)
            axs[i].set_xticks(ticks)
            axs[i].set_yticks(ticks)
            axs[i].set_aspect(1 / axs[i].get_data_ratio())
        
        # smoothed histogram of representativeness, normalized by number of frames 
        elif i == 1:
            # histogram plot
            sns.kdeplot(res.r_auto / res.n_full, fill=True, linewidth=1, 
                        ax=axs[i])
            sns.kdeplot(res.r_true / res.n_full, fill=True, linewidth=1, 
                        ax=axs[i])
            
            # text labels
            axs[i].text(0.885, 10, 'Automated', fontsize=8)
            axs[i].text(0.74, 5.8, 'Manual', fontsize=8)
            
            # axis formatting
            axs[i].set_xlabel('Representativeness', fontsize=12)
            axs[i].set_ylabel('Density', fontsize=12)
            axs[i].tick_params(labelsize=10)
            axs[i].set_xticks([0.7, 0.8, 0.9, 1.0])
            axs[i].set_yticks([0, 3, 6, 9, 12])
            axs[i].set_aspect(1 / axs[i].get_data_ratio())
            
        # smoothed histogram of uniqueness
        else:
            # histogram plot
            sns.kdeplot(res.u_auto / res.n_auto, fill=True, linewidth=1, 
                        ax=axs[i])
            sns.kdeplot(res.u_true / res.n_true, fill=True, linewidth=1, 
                        ax=axs[i])
            
            # text labels
            axs[i].text(0.02, 2.5, 'Automated', fontsize=8)
            axs[i].text(0.5, 1.8, 'Manual', fontsize=8)
            
            # axis formatting
            axs[i].set_xlabel('Uniqueness', fontsize=12)
            axs[i].set_ylabel('Density', fontsize=12)
            axs[i].set_xlim(0, 0.74)
            axs[i].tick_params(labelsize=10)
            axs[i].set_xticks([0.0, 0.2, 0.4, 0.6])
            axs[i].set_yticks([0, 1, 2, 3])
            axs[i].set_aspect(1 / axs[i].get_data_ratio())
    
    # layout and save
    if not constrained_layout: fig.tight_layout()
    plt.savefig(join(ROOT, 'results', 'figs', 'figureS7-4.pdf'), dpi=200, bbox_inches='tight')
    
if __name__ == '__main__':
    main()