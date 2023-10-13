import os
import pandas as pd
import logging
from datetime import datetime
import zstandard
import json

from common import constants, db_config, directories

logger = logging.getLogger(constants.LOGGER_NAME)


class RedditDumpLoader:
    def __init__(self, file_path: str, topic: str, platform: str):
        self.file_path = file_path
        self.topic = topic
        self.platform = platform
        self.file_names = [
            file for file in os.listdir(self.file_path) if file.endswith(".zst")
        ]
        logger.info(f"Found {len(self.file_names)} files in {self.file_path}")

        self.db = db_config.Database()
        # columns = constants.UNIFIED_COLUMNS[self.topic]
        columns = [
            "id TEXT PRIMARY KEY",
            "author_id TEXT NOT NULL",
            "date DATE",
            "title TEXT",
            "body TEXT",
            "interaction INTEGER",
            "subreddit TEXT",
            "subreddit_id TEXT",
            "matched_keyword TEXT",
        ]
        self.db.create_table(
            schema_name=self.topic, table_name=self.platform, columns=columns
        )

    def read_and_decode(
        self, reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0
    ):
        chunk = reader.read(chunk_size)
        bytes_read += chunk_size
        if previous_chunk is not None:
            chunk = previous_chunk + chunk
        try:
            return chunk.decode()
        except UnicodeDecodeError:
            if bytes_read > max_window_size:
                raise UnicodeError(
                    f"Unable to decode frame after reading {bytes_read:,} bytes"
                )
            logger.info(
                f"Decoding error with {bytes_read:,} bytes, reading another chunk"
            )
            return self.read_and_decode(
                reader, chunk_size, max_window_size, chunk, bytes_read
            )

    def load_data(self, file_name):
        with open(file_name, "rb") as file_handle:
            buffer = ""
            reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(
                file_handle
            )
            while True:
                chunk = self.read_and_decode(reader, 2**27, (2**29) * 2)

                if not chunk:
                    break
                lines = (buffer + chunk).split("\n")

                for line in lines[:-1]:
                    yield line.strip(), file_handle.tell()

                buffer = lines[-1]

            reader.close()

    def transform_data(self):
        chunk_size = 1000

        # For submissions the body is in the "selftext" field, for comments it's in the "body" field
        fields = ["title", "selftext"]
        values = constants.VACCINE_KEYWORDS
        logger.info(f"Searching for {values} in {fields}")
        exact_match = False

        # set data range
        start_date = datetime.strptime(
            constants.DATE_RANGE[self.topic]["start"], "%Y-%m-%d"
        ).date()
        end_date = datetime.strptime(
            constants.DATE_RANGE[self.topic]["end"], "%Y-%m-%d"
        ).date()

        for file_name in self.file_names:
            created = None
            matched_lines = 0
            bad_lines = 0
            total_lines = 0
            matched = False
            submissions_to_insert = []
            file_path_test = os.path.join(self.file_path, file_name)
            file_size = os.stat(file_path_test).st_size
            logger.info("-" * 100)
            logger.info(f"Processing {file_path_test} ({file_size:,} bytes)")

            for line, file_bytes_processed in self.load_data(file_path_test):
                total_lines += 1
                if total_lines % 100000 == 0:
                    logger.info(
                        f"{total_lines:,} : {matched_lines:,} : {bad_lines:,} : {file_bytes_processed:,}:{(file_bytes_processed / file_size) * 100:.0f}%"
                    )
                try:
                    obj = json.loads(line)
                    created = datetime.utcfromtimestamp(int(obj["created_utc"])).date()

                    if created < start_date or created > end_date:
                        continue

                    if not obj.get("id") or not obj.get("author_fullname"):
                        bad_lines += 1
                        continue

                    matched_keyword = None
                    # check if any of the fields contain the values
                    for field in fields:
                        field_value = obj[field].lower()
                        matched = False
                        for value in values:
                            if exact_match:
                                if field_value == value:
                                    matched = True
                                    matched_keyword = value
                                    break
                            else:
                                if value in field_value:
                                    matched = True
                                    matched_keyword = value
                                    break
                        if matched:
                            break

                    if matched:
                        matched_lines += 1
                        # Extract and convert required attributes
                        score = int(obj.get("score", 0))
                        num_comments = int(obj.get("num_comments", 0))

                        # Sum up score and num_comments for interaction
                        interaction = score + num_comments

                        submission_dict = {
                            "id": obj.get("id", ""),
                            "author_id": obj.get("author_fullname", ""),
                            "date": created,
                            "title": obj.get("title", ""),
                            "body": obj.get("selftext", ""),
                            "interaction": interaction,
                            "subreddit": obj.get("subreddit", ""),
                            "subreddit_id": obj.get("subreddit_id", ""),
                            "matched_keyword": matched_keyword,
                        }

                        submissions_to_insert.append(submission_dict)
                        # If list reaches chunk size, insert to database and reset list
                        if len(submissions_to_insert) == chunk_size:
                            self.insert_submissions_chunk_to_db(submissions_to_insert)
                            submissions_to_insert = []

                        # save the extracted data as a json file in directories.REDDIT_DUMP_DATA
                        # file_name = f"{obj['id']}.json"
                        # file_path = os.path.join(directories.REDDIT_DUMP_DIR, file_name)
                        # with open(file_path, "w") as file_handle:
                        #     json.dump(submission_dict, file_handle)

                        matched = False  # Reset for next iteration

                except Exception as e:
                    logger.error(f"Error processing line: {e}")

            # After the file is read, there might be some submissions left in the list
            # that haven't been saved to the database.
            if submissions_to_insert:
                self.insert_submissions_chunk_to_db(submissions_to_insert)

            logger.info(
                f"Finished processing {total_lines:,} lines, matched {matched_lines:,} lines, skipped {bad_lines:,} lines"
            )

    def insert_submissions_chunk_to_db(self, submissions):
        """Insert a chunk of submissions to the database."""
        df = pd.DataFrame(submissions)

        try:
            schema_name = self.topic
            table_name = self.platform  # should be reddit
            self.db.insert_data(
                schema_name=schema_name, table_name=table_name, chunk=df
            )
        except Exception as e:
            logger.error(f"Error inserting submissions chunk to database: {e}")
