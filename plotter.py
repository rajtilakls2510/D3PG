import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from cycler import cycler
import math, os


class Plotter:
    # This class is used to plot the metrics that are/were being tracked during training or evaluation.
    # Simply initialize this class with the list of metrics that you want to track with appropriate paths
    # and call the show() method. This will bring up a matplotlib figure where you will be able to see graphs
    # for all the metrics that are being tracked.

    def __init__(self, actor_ids=None, log_path="", track_logs=None, frequency=5000, smoothing=0.8, name="Figure1"):
        if track_logs is None:
            track_logs = []
        if actor_ids is None:
            actor_ids = []
        self.actors = actor_ids
        self.log_path = log_path
        self.track_logs = track_logs
        self.frequency = frequency
        self.cols = 3
        self.rows = math.ceil(len(actor_ids)*len(track_logs) / self.cols)
        self.smoothingWeight = smoothing
        self.name = name

        # Setting up plot styles
        SMALL_SIZE = 10
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 14

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

        plt.rcParams["lines.color"] = "#F8F8F2"
        plt.rcParams["patch.edgecolor"] = "#F8F8F2"

        plt.rcParams["text.color"] = "#F8F8F2"

        plt.rcParams["axes.facecolor"] = "#282A36"
        plt.rcParams["axes.edgecolor"] = "#F8F8F2"
        plt.rcParams["axes.labelcolor"] = "#F8F8F2"

        plt.rcParams["axes.prop_cycle"] = cycler('color',
                                                 ['#8be9fd', '#ff79c6', '#50fa7b', '#bd93f9', '#ffb86c', '#ff5555',
                                                  '#f1fa8c', '#6272a4'])

        plt.rcParams["xtick.color"] = "#F8F8F2"
        plt.rcParams["ytick.color"] = "#F8F8F2"

        plt.rcParams["legend.framealpha"] = 0.9
        plt.rcParams["legend.edgecolor"] = "#44475A"

        plt.rcParams["grid.color"] = "#F8F8F2"

        plt.rcParams["figure.facecolor"] = "#383A59"
        plt.rcParams["figure.edgecolor"] = "#383A59"

        plt.rcParams["savefig.facecolor"] = "#383A59"
        plt.rcParams["savefig.edgecolor"] = "#383A59"

        # Boxplots
        plt.rcParams["boxplot.boxprops.color"] = "F8F8F2"
        plt.rcParams["boxplot.capprops.color"] = "F8F8F2"
        plt.rcParams["boxplot.flierprops.color"] = "F8F8F2"
        plt.rcParams["boxplot.flierprops.markeredgecolor"] = "F8F8F2"
        plt.rcParams["boxplot.whiskerprops.color"] = "F8F8F2"

        self.colors = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
        self.colors = self.colors * (len(self.actors)*len(self.track_logs) // len(self.colors) + 1)
        self.fig, self.axes = plt.subplots(nrows=self.rows, ncols=self.cols, num=self.name)
        if (len(self.actors)*len(self.track_logs)) % self.cols > 0:
            for i in range(self.cols - (len(self.actors)*len(self.track_logs)) % self.cols):
                self.fig.delaxes(self.axes[-1][self.cols - i - 1])
        plt.subplots_adjust(left=0.05,
                            bottom=0.1,
                            right=0.95,
                            top=0.95,
                            wspace=0.2,
                            hspace=0.3)

    def load_plot_data(self, actor_id, log):
        data = pd.read_csv(os.path.join(self.log_path, f"{actor_id}.csv"))
        return data.groupby(["train_step"], as_index=False).mean()[["train_step", log]], actor_id, log

    def plot(self, f):
        data = [self.load_plot_data(actor, log) for actor in self.actors for log in self.track_logs]

        for ax, (d, actor_id, log), color in zip(self.axes.flat, data, self.colors):
            xlabel, ylabel = d.keys()
            smoothed = []
            try:
                last = d[ylabel][0]
                for smoothee in d[ylabel]:
                    last = last * self.smoothingWeight + (1 - self.smoothingWeight) * smoothee
                    smoothed.append(last)
            except:
                pass
            ax.clear()
            ax.plot(d[xlabel], d[ylabel], color=color, linewidth=0.5, alpha=0.25)  # Original Plot
            ax.plot(d[xlabel], smoothed, color=color, linewidth=1)  # Smoothed Plot
            ax.set_title(f"Actor: {actor_id} {log}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(visible=True, linewidth=0.05)

    def show(self, block=True):
        anim = FuncAnimation(self.fig, self.plot, interval=self.frequency)
        plt.show(block=block)

if __name__ == "__main__":
    plotter = Plotter(actor_ids=[639, 1076, 1976], log_path=os.path.join("mountain_car_cont_agent2", "actor_logs"), track_logs=["EpisodeLength", "TotalReward", "EpisodeTime"])
    plotter.show()