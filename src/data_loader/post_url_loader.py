import os
import re
import requests
import pandas as pd
import logging

from psycopg2 import sql
from tqdm import tqdm

from common import constants, db_config, directories
from .base_data_loader import BaseDataLoader

logger = logging.getLogger(constants.LOGGER_NAME)

UNVALID_URLS = [
    "facebook.com",
    "reddit.com",
    "twitter.com",
    "youtube.com",
    "instagram.com",
    "t.me",
    "tiktok.com",
    "PARSE",
    "ERROR",
    "linkedin.com",
    "pinterest.com",
    "tumblr.com",
    "snapchat.com",
    "whatsapp.com",
    "messenger.com",
    "discord.com",
    "viber.com",
    "telegram.org",
    "vk.com",
    "weibo.com",
    "line.me",
    "wechat.com",
    "kakaotalk.com",
    "qq.com",
    "signal.org",
]


class PostUrlLoader:
    def __init__(self, platform, file_path=None):
        self.chunk_size = 10000
        self.platform = platform
        self.file_path = file_path
        self.topic = "chatgpt"
        self.schema = "chatgpt"
        self.data = None

        self.db = db_config.Database()
        self.db.create_table(
            schema_name=self.schema,
            table_name="post_url",
            columns=constants.UNIFIED_COLUMNS["post_url"],
        )

    def process_urls(self):
        first_run = False
        if first_run:
            self.db.create_table(
                schema_name=self.schema,
                table_name="unvalid_urls",
                columns=["url TEXT PRIMARY KEY NOT NULL"],
            )
            for url in UNVALID_URLS:
                insert_query = sql.SQL(
                    """
                    INSERT INTO chatgpt.unvalid_urls (url) VALUES (%s);
                """
                )
                self.db.cursor.execute(insert_query, (url,))
                self.db.connection.commit()

        if self.platform == "facebook" or self.platform == "reddit":
            self.process_fb_rd()
        else:
            self.process_tw_ig_yt_urls()

    def load_data(self):
        """Load data from a given file path."""
        column_mapping = constants.POST_URL_COLUMN_CONFIG[self.platform]
        try:
            self.data = pd.read_csv(
                self.file_path, usecols=column_mapping.keys(), low_memory=False
            )
            self.data = self.data.rename(columns=column_mapping)
            logger.info(
                f"Successfully loaded {self.platform} data from {self.file_path}"
            )
        except Exception as e:
            logger.error(f"Error loading {self.platform} data: {e}")

    def load_twitter_data(self):
        dataframes = []
        self.file_names = [f for f in os.listdir(self.file_path) if f.endswith(".csv")]
        logger.info(f"Found {len(self.file_names)} files in {self.file_path}")
        total_files = len(self.file_names)
        for i, file in enumerate(self.file_names):
            logger.info(f"Loading file {i+1}/{total_files}")
            full_path = f"{self.file_path}/{file}"
            try:
                df = self.load_single_tw_file(full_path)
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Error loading {self.platform} data: {e}")
                continue
        self.data = pd.concat(dataframes, ignore_index=True)
        self.data = self.data.drop_duplicates(subset=["id"]).reset_index(drop=True)
        logger.info(f"Total records: {len(self.data)}")
        logger.info(f"Successfully loaded {self.platform} data from {self.file_path}")

    def load_single_tw_file(self, file_path: str):
        column_mapping = constants.POST_URL_COLUMN_CONFIG[self.platform]
        dtypes = {"id": str}
        df = pd.read_csv(
            file_path, usecols=column_mapping.keys(), low_memory=False, dtype=dtypes
        )
        df.rename(columns=column_mapping, inplace=True)
        return df

    def load_post_url(self):
        # Get all 'id' values from the cleaned full data table
        query = sql.SQL(
            """
            SELECT id
            FROM {}.{}
            """
        ).format(sql.Identifier(self.topic), sql.Identifier(self.platform))

        self.db.cursor.execute(query)
        result = self.db.cursor.fetchall()
        # Convert result to a set for faster search
        db_ids = {item[0] for item in result}

        if self.platform == "twitter":
            self.load_twitter_data()
        else:
            self.load_data()
        # Update the dataframe to retain only rows where 'id' is in db_ids
        self.data = self.data[self.data["id"].isin(db_ids)]
        logger.info(f"Filtered {self.platform} data by id")

        self.db.create_table(
            schema_name=self.schema,
            table_name=f"{self.platform}_url",
            columns=constants.PLATFORM_URL_TABLE_COLUMNS[self.platform],
        )

        if self.platform == "facebook" or self.platform == "reddit":
            self.insert_fb_rd_urls()
        elif self.platform == "twitter":
            self.insert_twitter_urls()
        elif self.platform == "youtube" or self.platform == "instagram":
            self.insert_yt_ig_urls()

    def insert_yt_ig_urls(self):
        load_from_saved = True
        if load_from_saved:
            file_path = f"{directories.PLATFORM_NEWS_DIR}/{self.platform}_gdelt.csv"
            df_gdelt_posts = pd.read_csv(file_path)
            df_gdelt_posts = df_gdelt_posts[["id", "url"]]
            self.data = df_gdelt_posts.dropna(subset=["id", "url"])
            self.data = self.data.rename(columns={"url": "target_url"})
            logger.info(f"Loaded {len(self.data)} posts from {file_path}")
        else:
            self.data = self._process_instagram_youTube_urls()

        self.save_to_db()

    def insert_fb_rd_urls(self):
        self.data = self.data.dropna(subset=["target_url", "expanded_url"])
        logger.info(
            f"Number of rows after dropping null target_url and expanded_url: {len(self.data)}"
        )

        # insert the filtered data into the table chatgpt.platform_url
        self.save_to_db()

    def insert_twitter_urls(self):
        self.data = self.data.dropna(subset=["target_url"])
        logger.info(f"Number of rows after dropping null target_url: {len(self.data)}")

        # target_url in twitter data is a string of urls separated by space
        # split the string into a list of urls and create a new row for each url
        self.data = self.data.assign(
            target_url=self.data["target_url"].str.split(" ")
        ).explode("target_url")

        # filter none values
        self.data = self.data[self.data["target_url"] != "None"]
        self.data = self.data[self.data["target_url"] != "none"]

        logger.info(f"Number of rows after dropping None: {len(self.data)}")

        # insert the filtered data into the table chatgpt.platform_url
        self.save_to_db()

    def check_gdelt_table(self):
        pass

    def base_process_urls(self):
        index_query_url = sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS idx_{}_url_id ON chatgpt.{}(id);
            """
        ).format(sql.Identifier(self.platform), sql.Identifier(f"{self.platform}_url"))
        self.db.cursor.execute(index_query_url)
        self.db.connection.commit()
        logger.info(f"Created index on {self.platform}_url table")

        index_query = sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS idx_{}_id ON chatgpt.{}(id);
            """
        ).format(sql.Identifier(self.platform), sql.Identifier(self.platform))
        self.db.cursor.execute(index_query)
        self.db.connection.commit()
        logger.info(f"Created index on {self.platform} table")

        # Drop all rows where id doesn't have a topic_name in chatgpt.{platform} table
        delete_query = sql.SQL(
            """
        DELETE FROM chatgpt.{platform_url}
        USING chatgpt.{platform}
        WHERE chatgpt.{platform_url}.id = chatgpt.{platform}.id
        AND chatgpt.{platform}.topic_name IS NULL;
        """
        ).format(
            platform_url=sql.Identifier(f"{self.platform}_url"),
            platform=sql.Identifier(self.platform),
        )
        self.db.cursor.execute(delete_query)
        self.db.connection.commit()
        logger.info(f"Deleted rows with no topic_name")

        # Drop all rows where target_url domain is in UNVALID_URLS
        delete_query = sql.SQL(
            """
            DELETE FROM chatgpt.{}
            WHERE target_url IN (SELECT url FROM chatgpt.unvalid_urls);
        """
        ).format(sql.Identifier(f"{self.platform}_url"))
        self.db.cursor.execute(delete_query)
        self.db.connection.commit()
        logger.info(f"Deleted rows with unvalid target_url")

    def process_tw_ig_yt_urls(self):
        logger.info(f"Processing {self.platform} URLs")
        if self.platform != "twitter":
            self.base_process_urls()

        # Loop over the remaining rows of table chatgpt.platform_url
        get_rows_query = sql.SQL(
            """
            SELECT id, target_url
            FROM chatgpt.{}
            WHERE target_url IS NOT NULL;
            """
        ).format(sql.Identifier(f"{self.platform}_url"))
        self.db.cursor.execute(get_rows_query)
        data_url = self.db.cursor.fetchall()
        logger.info(f"Number of rows to process: {len(data_url)}")

        for row in tqdm(data_url, total=len(data_url)):
            id_, target_url = row

            # Get topic_name from chatgpt.{platform} table based on id
            get_topic_query = sql.SQL(
                """
                SELECT topic_name
                FROM chatgpt.{}
                WHERE id = %s
                """
            ).format(sql.Identifier(self.platform))
            self.db.cursor.execute(get_topic_query, (id_,))
            topic_name = self.db.cursor.fetchone()[0]

            if not topic_name:
                continue

            # Check if the target_url is found in the GDELT database
            filter_query = sql.SQL(
                """
                SELECT url, tone
                FROM chatgpt.gdelt
                WHERE url = %s
                """
            )
            self.db.cursor.execute(filter_query, (target_url,))
            gdelt_row = self.db.cursor.fetchone()

            # If yes, update then insert the row into the post_url table
            if gdelt_row:
                url, tone = gdelt_row
                insert_query = sql.SQL(
                    """
                    INSERT INTO chatgpt.post_url (id, url, tone, topic, platform) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING;
                """
                )
                self.db.cursor.execute(
                    insert_query, (id_, url, tone, topic_name, self.platform)
                )
                self.db.connection.commit()

            # if none of the above, continue with the next row
            else:
                continue

        self.db.close()
        logger.info(f"Data insertion completed")

    def process_fb_rd(self):
        logger.info(f"Processing {self.platform} URLs")
        self.base_process_urls()

        # Loop over the remaining rows of table chatgpt.platform_url
        get_rows_query = sql.SQL(
            """
            SELECT id, target_url, expanded_url
            FROM chatgpt.{}
            WHERE target_url IS NOT NULL OR expanded_url IS NOT NULL;
            """
        ).format(sql.Identifier(f"{self.platform}_url"))
        self.db.cursor.execute(get_rows_query)
        data_url = self.db.cursor.fetchall()
        logger.info(f"Number of rows to process: {len(data_url)}")

        for row in tqdm(data_url, total=len(data_url)):
            id_, target_url, expanded_url = row

            # 2. Check if the target_url is found in the GDELT database
            filter_query = sql.SQL(
                """
                SELECT url, tone
                FROM chatgpt.gdelt
                WHERE url = %s
                """
            )
            self.db.cursor.execute(filter_query, (target_url,))
            gdelt_row = self.db.cursor.fetchone()

            # Get topic_name from chatgpt.{platform} table based on id
            get_topic_query = sql.SQL(
                """
                SELECT topic_name
                FROM chatgpt.{}
                WHERE id = %s
                """
            ).format(sql.Identifier(self.platform))
            self.db.cursor.execute(get_topic_query, (id_,))
            topic_name = self.db.cursor.fetchone()[0]

            if not topic_name:
                continue

            # 3. If yes, update then insert the row into the post_url table
            if gdelt_row:
                url, tone = gdelt_row
                insert_query = sql.SQL(
                    """
                    INSERT INTO chatgpt.post_url (id, url, tone, topic, platform) VALUES (%s, %s, %s, %s, %s);
                """
                )
                self.db.cursor.execute(
                    insert_query, (id_, url, tone, topic_name, self.platform)
                )
                self.db.connection.commit()

            # 4. If no, check the expanded_url
            elif expanded_url:
                self.db.cursor.execute(filter_query, (expanded_url,))
                gdelt_row = self.db.cursor.fetchone()
                if gdelt_row:
                    url, tone = gdelt_row
                    insert_query = sql.SQL(
                        """
                        INSERT INTO chatgpt.post_url (id, url, tone, topic, platform) VALUES (%s, %s, %s, %s, %s);
                    """
                    )
                    self.db.cursor.execute(
                        insert_query, (id_, url, tone, topic_name, self.platform)
                    )
                    self.db.connection.commit()

            # 5. if none of the above, continue with the next row
            else:
                continue

        self.db.close()
        logger.info(f"Data insertion completed")

    def get_chunks(self):
        """Yield successive n-sized chunks from the dataframe."""
        for i in range(0, len(self.data), self.chunk_size):
            yield self.data[i : i + self.chunk_size]

    def save_to_db(self):
        chunks = self.get_chunks()
        total_chunks = (len(self.data) - 1) // self.chunk_size + 1
        logger.info(f"Total number of chunks: {total_chunks}")
        for chunk_index, chunk in enumerate(chunks):
            try:
                self.db.insert_data(
                    schema_name=self.schema,
                    table_name=f"{self.platform}_url",
                    chunk=chunk,
                )
                logger.info(f"Inserted chunk {chunk_index + 1}/{total_chunks}")
            except Exception as e:
                first_row_id = chunk.iloc[0]["id"]
                logger.error(
                    f"Error inserting batch {chunk_index + 1} starting with row id {first_row_id}: {e}"
                )
        self.db.close()
        logger.info(f"Data insertion completed")

    # Instagram and YouTube URLs are not processed
    def _find_urls(self, text):
        url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        urls = re.findall(url_pattern, text)
        return urls

    def _expand_urls(self, url, timeout=2):
        session = requests.Session()
        try:
            response = session.head(url, allow_redirects=True, timeout=timeout)
            expanded_url = response.url
            return expanded_url
        except requests.exceptions.Timeout:
            return "undefined"
        except Exception as e:
            # print(f"Error expanding URL: {e}")
            return "undefined"

    def _process_instagram_youTube_urls(self):
        logger.info(f"Method: _process_instagram_youTube_urls needs to be updated")

    #     """Process Instagram and YouTube URLs"""
    #     logger.info(f"Method: _process_instagram_youTube_urls")
    #     logger.info(f"Processing {self.platform} URLs")
    #     posts_in_gdelt = []
    #     data_size = self.data.shape[0]

    #     for i, (_, r) in enumerate(self.data.iterrows()):
    #         urls = self._find_urls(r["text"])

    #         if i % 1000 == 0:
    #             logger.info(f"Processed {i} rows out of {data_size} rows")

    #         if not urls:
    #             continue

    #         for url in urls:
    #             try:
    #                 url_to_use = "undefined"

    #                 if url in self.gdelt_data.index:
    #                     url_to_use = url
    #                 else:
    #                     expanded_url = self._expand_urls(url)
    #                     if expanded_url in self.gdelt_data.index:
    #                         url_to_use = expanded_url

    #                 if url_to_use != "undefined":
    #                     post = {
    #                         "id": r["id"],
    #                         "date": r["date"],
    #                         "interactions": r["interaction"],
    #                         "url": url_to_use,
    #                         "url_tone": self.gdelt_data.loc[url_to_use, "tone"],
    #                     }
    #                     posts_in_gdelt.append(post)
    #             except Exception as e:
    #                 logger.error(f"Error processing row {r['id']}: {e}")
    #                 continue

    #     df_gdelt_posts = pd.DataFrame(posts_in_gdelt)
    #     df_gdelt_posts.to_csv(
    #         f"{directories.PLATFORM_NEWS_DIR}/{self.platform}_gdelt.csv", index=False
    #     )
    #     logger.info(
    #         f"Found {df_gdelt_posts.shape[0]} posts in the GDELT dataset for {self.platform}"
    #     )
