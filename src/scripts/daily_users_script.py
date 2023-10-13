from common import logging_config, constants
from analysis.daily_users import DailyUsers

logger = logging_config.setup_logger(constants.LOGGER_NAME)


def count_daily_users():
    for topic in constants.TOPIC_TABLES:
        if topic["schema_name"] != "covid":
            continue
        logger.info(f"Processing {topic['schema_name']} schema")
        for table in topic["table_name"]:
            logger.info(f"Processing {topic['schema_name']}.{table} table")
            daily_posts = DailyUsers(topic["schema_name"], table)
            daily_posts.process_daily_user_counts()
            logger.info("-" * 100)
        logger.info("=" * 100)


if __name__ == "__main__":
    count_daily_users()
