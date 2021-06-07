import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

# Adapted from https://stackoverflow.com/a/55262107
def bar_plot(
    ax, 
    data, 
    xticks,
    colors=None, 
    total_width=0.8,
    single_width=1, 
    legend=True
):
    """
    Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the 
        names of the data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    xticks : array-like
        A list of xticks names.
    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """
    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width/n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Set xticks
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_xticklabels(xticks)

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars/2)*bar_width + bar_width/2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(
                x + x_offset, 
                y, 
                width=bar_width*single_width, 
                color=colors[i%len(colors)]
            )

        # Add a handle to the last drawn bar for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys(), 
				  loc='best', fontsize='small',
				  framealpha=0.5)


def metrics_rand(n_rand, N, metrics, engines):
    """Extract metric information for 'n_rand' random queries."""
    
    id_rand = np.random.choice(np.arange(0, N), 
							   size=n_rand, 
							   replace=False)
    data = []

    for i in range(n_rand):
        dd = {}
        for j in range(len(engines)):
            mm = []
            for metric in metrics.values():
                mm.append(metric[id_rand[i], j])
            dd[engines[j]] = mm
        data.append(dd)

    return data, id_rand


def metrics_summary(functions, metrics, engines):
    """Generate summary statistics for a set of metrics."""

    data = []
    for f in functions:
        dd = {}
        mm = []

        for metric in metrics.values():
            mm.append(f(metric))
            
        for j in range(len(engines)):
            dd[engines[j]] = [mm[k][j] for k in range(len(metrics))]

        data.append(dd)

    return data
