import logging
import pandas as pd

from psycopg2 import sql
from common import constants, db_config, plotter

logger = logging.getLogger(constants.LOGGER_NAME)


class PostProcessTopics:
    def __init__(self, platform, topic):
        self.platform = platform
        self.topic = topic
        self.schema = "chatgpt"
        self.db = db_config.Database()

    def update_count(self):
        query = sql.SQL(
            """
            WITH topic_counts AS (
                SELECT topic, COUNT(*) as num
                FROM {schema}.{platform_topics}
                GROUP BY topic
            )
            UPDATE {schema}.{platform_topics_stats} AS stats
            SET count = topic_counts.num
            FROM topic_counts
            WHERE stats.topic = topic_counts.topic;
            """
        ).format(
            schema=sql.Identifier(self.schema),
            platform_topics=sql.Identifier(f"{self.platform}_topics"),
            platform_topics_stats=sql.Identifier(f"{self.platform}_topics_stats"),
        )

        self.db.cursor.execute(query)
        self.db.connection.commit()
        logger.info(f"Updated count for {self.platform} topic")

    def add_topic_name(self):
        # Add the topic_name column
        add_column_query = sql.SQL(
            """
            ALTER TABLE {schema}.{platform_topics}
            ADD COLUMN IF NOT EXISTS topic_name VARCHAR;
            """
        ).format(
            schema=sql.Identifier(self.schema),
            platform_topics=sql.Identifier(f"{self.platform}_topics"),
        )

        self.db.cursor.execute(add_column_query)

        # Update the topic_name column with data from general_topic column in topics_stats
        update_query = sql.SQL(
            """
            UPDATE {schema}.{platform_topics} AS topics
            SET topic_name = stats.general_topic
            FROM {schema}.{platform_topics_stats} AS stats
            WHERE topics.topic = stats.topic;
            """
        ).format(
            schema=sql.Identifier(self.schema),
            platform_topics=sql.Identifier(f"{self.platform}_topics"),
            platform_topics_stats=sql.Identifier(f"{self.platform}_topics_stats"),
        )

        self.db.cursor.execute(update_query)
        self.db.connection.commit()
        logger.info(f"Added topic_name column for {self.platform} topic")

    def create_summary_table(self):
        platforms = [
            "instagram",
            "youtube",
            "reddit",
            "facebook",
            "twitter",
        ]

        columns = [
            "topic_name VARCHAR NOT NULL",
            "count INT NOT NULL",
            "platform VARCHAR NOT NULL",
            "PRIMARY KEY (topic_name, platform)",
        ]

        self.db.create_table(
            schema_name="chatgpt",
            table_name="topics_stats_summary",
            columns=columns,
        )

        for platform in platforms:
            query = sql.SQL(
                """
                INSERT INTO chatgpt.topics_stats_summary (topic_name, count, platform)
                SELECT 
                    topic_name,
                    COUNT(*),
                    %s 
                FROM chatgpt.{}
                WHERE topic_name IS NOT NULL
                GROUP BY topic_name;                
                """
            ).format(
                sql.Identifier(platform),
            )

            self.db.cursor.execute(query, (platform,))
            self.db.connection.commit()
            logger.info(f"Added {platform} data to topics_summary table")

        # 1. Add the 'percentage' column to the topics_stats_summary table
        add_column_query = sql.SQL(
            """
            ALTER TABLE chatgpt.topics_stats_summary
            ADD COLUMN IF NOT EXISTS percentage FLOAT;
            """
        )

        self.db.cursor.execute(add_column_query)

        # 2. Calculate and update the percentages
        percentage_query = sql.SQL(
            """
            WITH TotalCounts AS (
                SELECT platform, SUM(count) AS total_count
                FROM chatgpt.topics_stats_summary
                GROUP BY platform
            )
            UPDATE chatgpt.topics_stats_summary AS summary
            SET percentage = (summary.count::FLOAT / totals.total_count) * 100
            FROM TotalCounts AS totals
            WHERE summary.platform = totals.platform;
            """
        )

        self.db.cursor.execute(percentage_query)
        self.db.connection.commit()

        logger.info("Updated topic percentages for each platform")
