"""
Plot related functions
"""
from config import cfg

import matplotlib as mpl
if not cfg.INTERACTIVE_PLOT:
    mpl.use('Agg')
import matplotlib.pyplot as plt


def plot_loss(exp_dir):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(11,8))

    max_iter = 0
    max_file = ''
    for loss_file in os.listdir(exp_dir):
        match = re.match(".+_loss_iter_([0-9]+).json", loss_file)
        if match is None:
            continue
        else:
            curr_iter = int(match.groups()[0])
            if curr_iter > max_iter:
                max_iter = curr_iter
                max_file = loss_file

    if max_iter == 0:
        print('No loss file found')

    with open(os.path.join(exp_dir, max_file)) as f:
        data = json.load(f)

    # Throw away first 20 data to see smaller scale more carefully
    data = data[20:]
    # plot the moving average as well
    plt.clf()
    gs = gridspec.GridSpec(1, 2, width_ratios=[4,1])
    plt.subplot(gs[0])

    if len(data) < 500:
        print('Not enough data')
        return

    # Plot moving average to smooth the stochastic outputs
    w = 20
    plt.plot(range(w - 1, len(data)),
             np.convolve(data, np.ones((w,))/w, mode= 'valid'), lw=1,
             label='MA-%d' % w)
    w = 500
    plt.plot(range(w - 1, len(data)),
             np.convolve(data, np.ones((w,))/w, mode= 'valid'), lw=4,
             label='MA-%d' % w)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid()
    plt.gcf().tight_layout()
    plt.gcf().savefig(os.path.join(exp_dir, 'loss_iter_%d.pdf' % max_iter))
    plt.gcf().savefig(os.path.join(exp_dir, 'loss_iter_%d.png' % max_iter))


def remove_tic():
    ax = plt.gca()
    ax.grid(True)
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
