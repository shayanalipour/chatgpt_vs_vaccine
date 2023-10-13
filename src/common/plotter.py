import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from common import directories, constants
from common.plot_config import (
    STYLE,
    CONTEXT,
    FONT_SCALE,
    AXIS_FONT_SIZE,
    TITLE_FONT_SIZE,
    FIG_SIZE,
    COLOR_PALETTE,
    COLOR_COUNT,
    PLATFORM_DISPLAY_NAMES,
    GENERAL_TOPIC_DISPLAY_NAMES,
    BOX_COLOR,
)

sns.set_style(STYLE)
sns.set_context(CONTEXT, font_scale=FONT_SCALE)


class Plotter:
    def __init__(self):
        self.axis_font_size = AXIS_FONT_SIZE
        self.title_font_size = TITLE_FONT_SIZE
        self.fig_size = FIG_SIZE
        self.colors = sns.color_palette(COLOR_PALETTE, COLOR_COUNT)
        self.platform_colors = self._assign_platform_colors(self.colors)

    @staticmethod
    def _assign_platform_colors(color_palette):
        platform_list = list(PLATFORM_DISPLAY_NAMES.values())
        return {
            platform: color for platform, color in zip(platform_list, color_palette)
        }

    @staticmethod
    def _replace_platform_names(data):
        """Replace platform names with their display names."""
        data.loc[:, "platform"] = data["platform"].replace(PLATFORM_DISPLAY_NAMES)

        return data

    @staticmethod
    def _replace_general_topic_names(data):
        """Replace general topic names with their display names."""
        data["topic_name"] = (
            data["topic_name"].replace(GENERAL_TOPIC_DISPLAY_NAMES).str.strip()
        )
        data = data.sort_values("topic_name")
        return data

    def plot_fig1(self, line_data, scatter_data, topic):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        fig.subplots_adjust(wspace=0.3, bottom=0.15)

        # Plot 1: Line plot
        line_data = self._replace_platform_names(line_data)
        ax1 = sns.lineplot(
            data=line_data,
            x="date",
            y="cumulative_posts",
            hue="platform",
            palette=self.platform_colors,
            linewidth=4,
            alpha=0.7,
            ax=ax1,
        )

        ax1.tick_params(axis="both", which="major", labelsize=self.axis_font_size)
        num_ticks = 6
        x_dates = sorted(
            line_data["date"].unique().tolist()
        )  # Ensuring the dates are sorted
        spacing = len(x_dates) // (num_ticks - 1)
        selected_ticks = [x_dates[0]]
        for i in range(1, num_ticks - 1):
            selected_ticks.append(x_dates[i * spacing])
        selected_ticks.append(x_dates[-1])
        ax1.set_xticks(selected_ticks)
        plt.setp(ax1.get_xticklabels(), rotation=30)
        ax1.set_xlabel("")
        ax1.set_ylabel("Cumulative Number of Posts", fontsize=self.axis_font_size)
        ax1.set(yscale="log")
        ax1.text(
            0.02,
            0.98,
            "a",
            fontsize=30,
            fontweight="bold",
            va="top",
            transform=ax1.transAxes,
        )

        handles, labels = ax1.get_legend_handles_labels()
        for handle in handles:
            handle.set_linewidth(4)
        ax1.legend(
            loc="lower right",
            fontsize=self.axis_font_size,
            markerscale=2,
            handles=handles,
        )

        # Plot 2: Scatter plot
        scatter_data = self._replace_platform_names(scatter_data)
        ax2 = sns.scatterplot(
            data=scatter_data,
            x="interaction_count",
            y="post_count",
            hue="platform",
            palette=self.platform_colors,
            s=80,
            alpha=0.5,
            linewidth=0.0,
            ax=ax2,
        )

        ax2.tick_params(axis="both", which="major", labelsize=self.axis_font_size)
        ax2.set_xlabel("Number of Interactions", fontsize=self.axis_font_size)
        ax2.set_ylabel("Number of Posts", fontsize=self.axis_font_size)
        ax2.set(xscale="log", yscale="log")
        ax2.text(
            0.02,
            0.98,
            "b",
            fontsize=30,
            fontweight="bold",
            va="top",
            transform=ax2.transAxes,
        )

        handles, labels = ax2.get_legend_handles_labels()
        for handle in handles:
            handle.set_linewidth(4)
        ax2.legend(
            loc="upper right",
            fontsize=self.axis_font_size,
            markerscale=2,
            handles=handles,
        )

        fig.savefig(
            f"{directories.PLOT_DIR}/combined_fig1_{topic}.pdf",
            format="pdf",
        )

    def logistic_model_plot(
        self,
        data,
        topic,
        platform,
        x_col="day",
        y_col="cumulative_unique_users",
        sigmoid_fit=None,
        chatgpt_data=None,
        chatgpt_sigmoid_fit=None,
        ax=None,
    ):
        if sigmoid_fit is None:
            raise ValueError("sigmoid_fit cannot be None")

        data = data.copy()
        data = self._replace_platform_names(data)
        platform_display_name = data.iloc[0]["platform"]

        # plt.figure(figsize=self.fig_size)

        # set figure title, x and y labels
        ax.set_title(
            platform_display_name,
            fontsize=self.title_font_size,
        )
        ax.set_xlabel("Days", fontsize=self.axis_font_size)
        # plt.xlabel("Days", fontsize=self.axis_font_size)
        if platform == "news":
            ax.set_ylabel(
                "Cumulative Number of News Articles", fontsize=self.axis_font_size
            )
        else:
            ax.set_ylabel("Cumulative Number of Users", fontsize=self.axis_font_size)

        # set x and y ticks
        # plt.xticks(fontsize=self.axis_font_size)
        # plt.yticks(fontsize=self.axis_font_size)
        ax.tick_params(axis="both", which="major", labelsize=self.axis_font_size)

        if chatgpt_data is not None:
            # plot the data
            ax.plot(
                data[x_col],
                data[y_col],
                linewidth=4,
                color=self.platform_colors[platform_display_name],
                alpha=0.8,
                label="Vaccine Data",
            )

            # plot the fitted sigmoid
            ax.plot(
                data[x_col],
                sigmoid_fit,
                linewidth=4,
                color="black",
                alpha=0.6,
                label="Vaccine Fit",
            )

        if topic == "covid":
            ax.plot(
                chatgpt_data[x_col],
                chatgpt_data[y_col],
                linewidth=4,
                linestyle="dashed",
                color=self.platform_colors[platform_display_name],
                alpha=0.8,
                label="ChatGPT Data",
            )

            ax.plot(
                chatgpt_data[x_col],
                chatgpt_sigmoid_fit,
                linewidth=4,
                linestyle="dashed",
                color="black",
                alpha=0.6,
                label="ChatGPT Fit",
            )

        else:
            # plot the data
            ax.plot(
                data[x_col],
                data[y_col],
                linewidth=8,
                color=self.platform_colors[platform_display_name],
                alpha=0.8,
                label="Data",
            )

            # plot the fitted sigmoid
            ax.plot(
                data[x_col],
                sigmoid_fit,
                linewidth=8,
                color="black",
                alpha=0.6,
                label="Fit",
            )

        # set legend
        ax.legend(loc="best", fontsize=self.axis_font_size)
        # # legend modification
        # ax.rcParams["legend.title_fontsize"] = "large"
        # ax.rcParams["legend.handlelength"] = 3.3
        # ax.rcParams["legend.labelspacing"] = 0.6
        # ax.rcParams["legend.borderpad"] = 0.3

        return ax

    def topic_modeling_sentiment_combined(self, heatmap_data, box_data, x_col, y_col):
        # Create a grid of 1 row and 2 columns, and set the size
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(30, 14))

        # Adjust the space between the plots
        fig.subplots_adjust(wspace=0.3)

        # Plot 1: Heatmap
        data_for_heatmap = self._replace_platform_names(heatmap_data)
        data_for_heatmap = self._replace_general_topic_names(data_for_heatmap)
        heatmap_data = data_for_heatmap.pivot(
            index="topic_name", columns="platform", values="percentage"
        )
        # Inside the heatmap section of your combined_plot function
        ax1 = sns.heatmap(
            data=heatmap_data,
            annot=True,
            cmap=sns.light_palette("seagreen", as_cmap=True),
            fmt=".2f",
            annot_kws={"size": 22, "color": "black"},
            cbar=False,
            ax=ax1,
        )
        ax1.set_aspect(0.40)
        ax1.set(xlabel="", ylabel="")
        ax1.tick_params(axis="both", colors="black", labelcolor="black", labelsize=20)
        ax1.text(-1, -1, "a", fontsize=30, fontweight="bold", va="top")

        # Plot 2: Box plot
        data_for_boxplot = self._replace_general_topic_names(box_data)
        ax2 = sns.boxplot(
            data=data_for_boxplot,
            x=x_col,
            y=y_col,
            color=BOX_COLOR,
            fliersize=0,
            ax=ax2,
        )
        for box in ax2.artists:
            box.set_edgecolor("black")
        for whisker in ax2.lines:
            whisker.set_color("black")
        lines = ax2.lines
        for i in range(4, len(lines), 6):
            lines[i].set_visible(False)
        sns.pointplot(
            data=data_for_boxplot,
            x=x_col,
            y=y_col,
            color="black",
            markers="d",
            scale=1.5,
            errorbar=None,
            linestyles="",
            estimator=np.mean,
            join=False,
            ax=ax2,
        )
        ax2.set_aspect(1.2)
        ax2.set_xlim(-8, 8)
        ax2.axvline(0, color="red", linestyle="--", linewidth=3, alpha=0.5)
        ax2.set_xlabel("Sentiment Tone", fontsize=20)
        ax2.set_ylabel("")
        ax2.tick_params(axis="both", which="major", labelsize=20)
        ax2.tick_params(axis="both", colors="black", labelcolor="black")
        ax2.text(-11, -1.5, "b", fontsize=30, fontweight="bold", va="top")

        fig.savefig(
            f"{directories.PLOT_DIR}/combined_plot.pdf",
            bbox_inches="tight",
            format="pdf",
        )
        fig.savefig(
            f"{directories.PLOT_DIR}/combined_plot.png",
            bbox_inches="tight",
            dpi=300,
            format="png",
        )

    ##########################################################
    ######################## OLD METHODS #####################
    ##########################################################

    def plot_counts(
        self,
        data,
        x_col,
        y_col,
        plot_type="line",
        legend_loc="best",
        x_label=None,
        y_label=None,
        x_scale=None,
        y_scale=None,
        **kwargs,
    ):
        data = self._replace_platform_names(data)
        fig, ax = plt.subplots(figsize=self.fig_size)

        if plot_type == "line":
            p = sns.lineplot(
                data=data,
                x=x_col,
                y=y_col,
                hue="platform",
                palette=self.platform_colors,
                linewidth=4,
                alpha=0.7,
                ax=ax,
                **kwargs,
            )
        elif plot_type == "scatter":
            p = sns.scatterplot(
                data=data,
                x=x_col,
                y=y_col,
                hue="platform",
                palette=self.platform_colors,
                s=80,
                alpha=0.5,
                linewidth=0.1,
                edgecolor="black",
                ax=ax,
                **kwargs,
            )

        ax.tick_params(axis="both", which="major", labelsize=self.axis_font_size)

        # x axis ticks modification
        num_ticks = 6
        x_dates = sorted(data[x_col].unique().tolist())  # Ensuring the dates are sorted

        # Calculate the desired spacing between ticks
        spacing = len(x_dates) // (num_ticks - 1)

        # Create a list to store the tick locations
        selected_ticks = [x_dates[0]]

        # Add ticks in the middle
        for i in range(1, num_ticks - 1):
            selected_ticks.append(x_dates[i * spacing])

        # Add the last date
        selected_ticks.append(x_dates[-1])

        ax.set_xticks(selected_ticks)
        plt.setp(ax.get_xticklabels(), rotation=30)

        # x and y axis labels
        if x_label:
            p.set_xlabel(x_label, fontsize=self.axis_font_size)
        else:
            p.set_xlabel(x_col, fontsize=self.axis_font_size)

        if y_label:
            p.set_ylabel(y_label, fontsize=self.axis_font_size)
        else:
            p.set_ylabel(y_col, fontsize=self.axis_font_size)

        # x and y axis scale
        if x_scale:
            p.set(xscale=x_scale)

        if y_scale:
            p.set(yscale=y_scale)

        # Legend modification
        handles, labels = ax.get_legend_handles_labels()
        for handle in handles:
            handle.set_linewidth(4)
        plt.legend(
            loc=legend_loc,
            fontsize=self.axis_font_size,
            markerscale=2,
            handles=handles,
        )

        return fig, ax, p

    def heatmap_topic_modeling(self, data):
        data = self._replace_platform_names(data)
        data = self._replace_general_topic_names(data)

        # pivot the data
        heatmap_data = data.pivot(
            index="topic_name", columns="platform", values="percentage"
        )

        plt.figure(figsize=self.fig_size)
        sns.heatmap(
            data=heatmap_data,
            annot=True,
            cmap=sns.light_palette("seagreen", as_cmap=True),
            fmt=".2f",
            annot_kws={"size": 16},
        )

        # remove x axis and y axis labels
        plt.xlabel("")
        plt.ylabel("")

        # Increase the font size of x and y axis labels
        ax = plt.gca()
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)

        # save the plot
        self.save_plot(plt, "heatmap_topic_modeling_oct11")

    def sentiment_box_plot(self, data, x_col, y_col):
        data = self._replace_general_topic_names(data)
        fig, ax = plt.subplots(figsize=self.fig_size)
        p = sns.boxplot(
            data=data,
            x=x_col,
            y=y_col,
            color=BOX_COLOR,
            fliersize=0,
            ax=ax,
        )

        # change box border to black
        for box in p.artists:
            box.set_edgecolor("black")

        # change whisker color to black
        for whisker in p.lines:
            whisker.set_color("black")

        # hid median line
        lines = ax.lines
        for i in range(4, len(lines), 6):
            lines[i].set_visible(False)

        # overlay a pointplot for the mean values
        sns.pointplot(
            data=data,
            x=x_col,
            y=y_col,
            color="black",
            markers="d",
            scale=1.5,
            errorbar=None,
            linestyles="",  # remove line connecting the markers
            estimator=np.mean,
            join=False,
            ax=ax,
        )

        # set x axis limit
        ax.set_xlim(-8, 8)

        # vertical line at 0
        p.axvline(0, color="red", linestyle="--", linewidth=3, alpha=0.5)

        # x and y axis labels
        p.set_xlabel("Sentiment Tone", fontsize=self.axis_font_size)
        p.set_ylabel("", fontsize=self.axis_font_size)

        p.tick_params(axis="both", which="major", labelsize=self.axis_font_size)

        # set the color of x and y axis ticks and labels to black
        ax.tick_params(axis="both", colors="black", labelcolor="black")

        # save the plot
        self.save_plot(plt, "sentiment_box_plot")

    def save_plot(self, fig, name):
        if not os.path.exists(directories.PLOT_DIR):
            os.makedirs(directories.PLOT_DIR)

        fig.savefig(f"{directories.PLOT_DIR}/{name}.png", bbox_inches="tight", dpi=300)
        fig.savefig(f"{directories.PLOT_DIR}/{name}.pdf", bbox_inches="tight")
