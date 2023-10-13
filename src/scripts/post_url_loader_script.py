import pandas as pd
from psycopg2 import sql
from common import logging_config, constants, db_config, directories
from common.plotter import Plotter
from data_loader.post_url_loader import PostUrlLoader

logger = logging_config.setup_logger(constants.LOGGER_NAME)

# This one is little messy, sorry about that!


def get_chunks(data, chunksize):
    """Yield successive n-sized chunks from the dataframe."""
    for i in range(0, len(data), chunksize):
        yield data[i : i + chunksize]


def load_gdelt_2db():
    chunksize = 50000
    news_df = pd.read_csv(directories.GDELT_FILE)
    news_df = news_df.drop_duplicates(subset=["url"])
    logger.info(f"Number of news articles: {len(news_df)}")

    db = db_config.Database()
    db.create_table(
        schema_name="chatgpt",
        table_name="gdelt",
        columns=[
            "url TEXT PRIMARY KEY NOT NULL",
            "tone REAL NOT NULL",
        ],
    )

    total_chunks = (len(news_df) - 1) // chunksize + 1
    chunks = get_chunks(news_df, chunksize)
    logger.info(f"Total number of chunks: {total_chunks}")
    for chunk_index, chunk in enumerate(chunks):
        try:
            db.insert_data(schema_name="chatgpt", table_name="gdelt", chunk=chunk)
            logger.info(f"Inserted chunk {chunk_index + 1}/{total_chunks}")
        except Exception as e:
            first_row_id = chunk.iloc[0]["id"]
            logger.error(
                f"Error inserting batch {chunk_index + 1} starting with row id {first_row_id}: {e}"
            )
    db.close()
    logger.info(f"Data insertion completed")


def load_platform_urls():
    platform = "youtube"
    if platform == "twitter":
        file_path = directories.RAW_TWIITER_DIR
    else:
        file_path = f"{directories.RAW_GPT_DIR}/{platform}.csv"

    post_url_loader = PostUrlLoader(platform=platform, file_path=file_path)
    post_url_loader.load_post_url()


def process_urls():
    platform = "twitter"
    post_url_loader = PostUrlLoader(platform=platform)
    post_url_loader.process_urls()


def sentiment_stats():
    db = db_config.Database()
    query = sql.SQL(
        """
        SELECT topic, tone
        FROM chatgpt.post_url
        """
    )
    db.cursor.execute(query)
    data = db.cursor.fetchall()
    data = pd.DataFrame(data, columns=["topic", "tone"])
    data_stats = _sentiment_stats(data)

    # create a new table to store the stats
    db.create_table(
        schema_name="chatgpt",
        table_name="topic_sentiment_stats",
        columns=[
            "topic TEXT PRIMARY KEY NOT NULL",
            "count INTEGER NOT NULL",
            "mean REAL NOT NULL",
            "std REAL NOT NULL",
            "Q1 REAL NOT NULL",
            "Q3 REAL NOT NULL",
            "min REAL NOT NULL",
            "max REAL NOT NULL",
        ],
    )

    db.insert_data(
        schema_name="chatgpt",
        table_name="topic_sentiment_stats",
        chunk=data_stats,
    )
    db.close()


def _sentiment_stats(data):
    tone_stats = data.groupby("topic")["tone"].agg(
        [
            "count",
            "mean",
            "std",
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75),
            "min",
            "max",
        ]
    )

    tone_stats.columns = ["count", "mean", "std", "Q1", "Q3", "min", "max"]
    tone_stats = tone_stats.sort_values(by="mean", ascending=False)
    tone_stats = tone_stats.reset_index()
    return tone_stats


if __name__ == "__main__":
    # load_gdelt_2db()
    # load_platform_urls()
    # process_urls()
    sentiment_stats()
