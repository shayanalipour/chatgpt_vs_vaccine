import logging
from common import constants, db_config
from psycopg2 import sql

logger = logging.getLogger(constants.LOGGER_NAME)


class InteractionDist:
    def __init__(self, schema_name: str, table_name: str):
        self.schema_name = schema_name
        self.table_name = table_name
        self.db = db_config.Database()

        self.db.create_table(
            "processed",
            "interaction_dist",
            constants.UNIFIED_COLUMNS["interaction_dist"],
        )

    def process_interaction_dist(self):
        query = sql.SQL(
            """
            SELECT 
            interaction, COUNT(*) 
            FROM {}.{} 
            GROUP BY interaction
            """
        ).format(sql.Identifier(self.schema_name), sql.Identifier(self.table_name))
        self.db.cursor.execute(query)
        result = self.db.cursor.fetchall()

        # Add platform and topic values to the result
        formatted_results = [
            (interaction, count, self.table_name, self.schema_name)
            for interaction, count in result
        ]

        # Insert the results in interaction_dist table
        insert_query = """
            INSERT INTO processed.interaction_dist (interaction_count, post_count, platform, topic)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (platform, topic, interaction_count) DO NOTHING;
        """
        self.db.cursor.executemany(insert_query, formatted_results)
        self.db.connection.commit()
        logger.info(f"Processed {self.table_name} interaction distribution.")
        self.db.close()
