import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def vis_scatter(df: pd.DataFrame):
    """
    Compute Scatter-Plot of output over input variable
    :param df: Pandas DataFrame
    :return: None
    """
    # melt input variables and maintain index variable dOx
    tmp = df.melt(id_vars=["dOx"], 
            var_name="input_variable", 
            value_name="value")

    sns.set_theme(style="ticks")

    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(tmp, col="input_variable", hue="input_variable", palette="tab20c", sharex=False, sharey=False,
                         col_wrap=3, height=4)

    # Draw a horizontal line at zero dOx
    grid.refline(y=0, linestyle=":")

    # Draw a scatter to show the trajectory of each random walk
    grid.map(plt.scatter, "value", "dOx", marker="o")

    # Adjust the arrangement of the plots
    grid.fig.tight_layout(w_pad=1)