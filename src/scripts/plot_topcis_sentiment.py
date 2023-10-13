import pandas as pd
from psycopg2 import sql

from common import constants, db_config, plotter


def plot_topic_sentiment():
    db = db_config.Database()

    # fecth data from topics_stats_summary table
    topic_query = sql.SQL(
        """
        SELECT topic_name, percentage, platform
        FROM chatgpt.topics_stats_summary
        ORDER BY topic_name;
        """
    )
    db.cursor.execute(topic_query)
    results = db.cursor.fetchall()
    summary_df = pd.DataFrame(results, columns=["topic_name", "percentage", "platform"])
    summary_df = summary_df.sort_values("topic_name")

    # fetch data from sentiments
    sentiment_query = sql.SQL(
        """
        SELECT topic, tone
        FROM chatgpt.post_url
        """
    )
    db.cursor.execute(sentiment_query)
    sent_data = db.cursor.fetchall()
    sent_data = pd.DataFrame(sent_data, columns=["topic", "tone"])

    sent_data = sent_data.rename(columns={"topic": "topic_name"})

    p = plotter.Plotter()
    p.topic_modeling_sentiment_combined(
        heatmap_data=summary_df,
        box_data=sent_data,
        x_col="tone",
        y_col="topic_name",
    )


if __name__ == "__main__":
    plot_topic_sentiment()
