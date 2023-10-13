import logging
import pandas as pd

from common import constants, db_config
from common.plotter import Plotter

logger = logging.getLogger(constants.LOGGER_NAME)


class GeneralCountsPlotter:
    """
    This class loads data from the database and uses the Plotter class to plot figure 1.
    General counts (figure 1) include: daily post count and interaction distribution.

    To use this class, run scripts/plot_general_counts_script.py
    """

    def __init__(self):
        self.db = db_config.Database()

    def load_data(self, table_name, columns):
        query = f"SELECT {', '.join(columns)} FROM {table_name};"
        self.db.cursor.execute(query)
        result = self.db.cursor.fetchall()
        data = pd.DataFrame(result, columns=columns)
        data = data.sort_values(["platform", "topic"])
        return data

    def load_daily_post_count(self):
        table_name = "processed.daily_post_counts"
        columns = ["date", "cumulative_posts", "platform", "topic"]
        data = self.load_data(table_name, columns)
        logger.info(f"Loaded {len(data)} rows from {table_name} table.")
        return data

    def load_interaction_dist(self):
        table_name = "processed.interaction_dist"
        columns = ["interaction_count", "post_count", "platform", "topic"]
        data = self.load_data(table_name, columns)
        logger.info(f"Loaded {len(data)} rows from {table_name} table.")
        return data

    def plot_fig1(self):
        line_data = self.load_daily_post_count()
        scatter_data = self.load_interaction_dist()

        for topic in line_data["topic"].unique():
            line_topic_data = line_data[line_data["topic"] == topic]
            scatter_topic_data = scatter_data[scatter_data["topic"] == topic]
            p = Plotter()
            p.plot_fig1(
                line_data=line_topic_data,
                scatter_data=scatter_topic_data,
                topic=topic,
            )
