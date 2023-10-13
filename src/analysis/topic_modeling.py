import logging
import numpy as np
import pandas as pd
from psycopg2 import sql
from tqdm import tqdm

import re
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModel
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

from common import constants, db_config
from common.plotter import Plotter

# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("averaged_perceptron_tagger")

tqdm.pandas(desc="Cleaning test progress")

logger = logging.getLogger(constants.LOGGER_NAME)


class TopicModeling:
    def __init__(self, platform: str, topic: str, detect_lang: bool = False):
        self.platform = platform
        self.topic = topic
        self.detect_lang = detect_lang
        self.data = None
        self.topics_df = None
        self.model = None
        self.db = db_config.Database()
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        # Load the language detection pipeline onto the GPU
        self.batch_size = 64
        self.lang_detect = pipeline(
            "text-classification",
            model="papluca/xlm-roberta-base-language-detection",
            tokenizer="papluca/xlm-roberta-base-language-detection",
            device=self.device,
            padding=True,
            truncation=True,
        )

        # Setup topic modeling model
        self.params = constants.TOPIC_MODELING_PARAMS[self.platform]
        logger.info(f"Model params: {self.params}")

        # Extract embeddings
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # To educe dimensionality
        self.umap_model = UMAP(
            n_neighbors=self.params["umap_neighbors"],
            n_components=self.params["umap_components"],
            metric="cosine",
        )

        # Cluster reduced embeddings
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=self.params["hdbscan_min_cluster_size"],
            min_samples=self.params["hdbscan_min_samples"],
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        # Initialize BERTopic model
        self.model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            n_gram_range=(1, 2),
            top_n_words=self.params["top_n_words"],
            low_memory=False,
        )

        if self.topic != "chatgpt":
            logger.error(
                f"Topic modeling is not supported for {self.topic}, try chatgpt"
            )
            raise NotImplementedError

    def run(self):
        self.get_data()
        self.add_clean_text_col()
        self.add_batch_col()
        if self.detect_lang:
            self.run_detect_language()
        self.run_topic_modeling()
        self.db.close()

    def run_topic_modeling(self):
        # select only english posts
        self.data = self.data[self.data["lang"] == "en"]
        # Remove stopwords and numbers from the text
        self.data["clean_text"] = self.data["text"].progress_apply(
            lambda x: self.clean_text(x, rm_stopwords=True)
        )
        self.data.dropna(subset=["clean_text"], inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        logger.info(f"Number of rows after filtering: {len(self.data)}")
        self.update_clean_text()

        batches = [self.data]  # By default, consider the entire data as a single batch

        if self.platform == "twitter":
            mid_idx = len(self.data) // 2
            batches = [self.data.iloc[:mid_idx, :], self.data.iloc[mid_idx:, :]]

        for idx, batch in enumerate(batches):
            docs = batch["clean_text"].tolist()
            current_batch = idx + 1
            logger.info(f"Processing batch {current_batch} with {len(docs)} docs")
            topics, _ = self.model.fit_transform(docs)
            logger.info(f"Number of topics: {len(self.model.get_topic_freq())}")
            new_topics = self.model.reduce_outliers(docs, topics)
            logger.info(
                f"Number of topics after reducing outliers: {len(set(new_topics))}"
            )
            self.model.update_topics(docs, topics=new_topics)

            topic_labels = self.model.generate_topic_labels(
                nr_words=5, topic_prefix=False, word_length=50, separator=" - "
            )
            self.topics_df = pd.DataFrame(self.model.get_topic_info())
            self.topics_df["topic_label"] = topic_labels
            self.topics_df.drop(columns=["Name"], inplace=True)
            self.topics_df.sort_values(by="Count", ascending=False, inplace=True)
            self.topics_df = self.topics_df[["Topic", "Count", "topic_label"]]
            self.topics_df.reset_index(drop=True, inplace=True)

            logger.info(f"Columns: {self.topics_df.columns}")
            logger.info(f"Topics: {self.topics_df.head()}")

            batch["topic"] = new_topics
            batch["batch"] = current_batch
            self.data = batch
            self.save_to_db(batch_num=current_batch)

    def run_detect_language(self):
        self.data.dropna(subset=["text"], inplace=True)
        self.data["text"] = self.data["text"].astype(str)
        self.data["clean_text"] = self.data["text"].apply(
            lambda x: self.clean_text(x, rm_stopwords=False)
        )
        self.data.dropna(subset=["clean_text"], inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        logger.info(f"Number of rows after cleaning: {len(self.data)}")

        logger.info("Detecting language...")
        texts = self.data["clean_text"].tolist()
        langs = []
        for i, batch_texts in enumerate(self.batch(texts, self.batch_size)):
            langs.extend(self.detect_language(batch_texts))
            if i % 25 == 0 and i > 0:
                logger.info(
                    f"processed {i * self.batch_size} rows - total {len(texts)}"
                )

        self.data["lang"] = langs
        self.data.dropna(subset=["lang"], inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        logger.info(f"Number of rows after language detection: {len(self.data)}")

        self.update_clean_text()

        # count the number of rows for each language
        lang_counts = self.data["lang"].value_counts()
        logger.info(f"Language counts: {lang_counts}")

    def get_data(self):
        query = sql.SQL(
            """
            SELECT * FROM {topic}.{platform}
            """
        ).format(
            topic=sql.Identifier(self.topic), platform=sql.Identifier(self.platform)
        )

        # Load the result into a DataFrame
        self.db.cursor.execute(query)
        result = self.db.cursor.fetchall()
        columns = [item.split()[0] for item in constants.UNIFIED_COLUMNS[self.topic]]
        self.data = pd.DataFrame(result, columns=columns)
        logger.info(f"Loaded {len(self.data)} rows from {self.topic}.{self.platform}")
        logger.info(f"Columns: {self.data.columns}")

    def get_wordnet_pos(self, tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def clean_text_base(self, text):
        """Base cleaning of the text, which is common to both tasks"""

        # Remove emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251"
            "]+"
        )
        text = emoji_pattern.sub(r"", text)

        # Remove URLs, HTML tags, and extra whitespaces
        text = re.sub(
            r"http\S+|www\S+|https\S+|<.*?>|\s+", " ", text, flags=re.MULTILINE
        )

        # Remove string if it only contains punctuation
        if text.strip(string.punctuation).strip() == "":
            return None

        # Remove \r and \n
        text = re.sub(r"\r|\n", "", text)

        # Remove 'x200b' and 'x200B' occurrences
        text = text.replace("x200b", "").replace("x200B", "")

        return text

    def lemmatize_text(self, text):
        """Tokenize, POS-tag and lemmatize the text"""

        # Tokenize and POS tag the text
        text = nltk.word_tokenize(text)
        pos_tagged_text = nltk.pos_tag(text)

        # Lemmatize the text using POS tags
        text = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos))
            for word, pos in pos_tagged_text
        ]

        return text

    def remove_stopwords_and_numbers(self, text):
        """Remove stopwords and numbers from the text"""

        return [
            word
            for word in text
            if word.lower() not in self.stop_words and not word.isdigit()
        ]

    def finalize_text(self, text):
        """Final steps in processing"""

        # Return None if the length of text is less than 2
        if len(text) < 2:
            return None

        # Truncate text to 1024 words
        text = text[:1024]

        # Join the words and convert to lowercase
        text = " ".join(text).lower()

        return text

    def clean_text(self, text, rm_stopwords=False):
        text = self.clean_text_base(text)

        if text is None:
            return None

        # Remove punctuation
        text = "".join(ch for ch in text if ch not in string.punctuation)

        text = self.lemmatize_text(text)

        if rm_stopwords:
            text = self.remove_stopwords_and_numbers(text)

        return self.finalize_text(text)

    def add_batch_col(self):
        add_batch_query = sql.SQL(
            """
            ALTER TABLE {topic}.{platform}
            ADD COLUMN IF NOT EXISTS batch INTEGER
            """
        ).format(
            topic=sql.Identifier(self.topic), platform=sql.Identifier(self.platform)
        )

        self.db.cursor.execute(add_batch_query)
        self.db.connection.commit()
        logger.info(f"Added batch column to {self.topic}.{self.platform}")

    def add_clean_text_col(self):
        add_clean_text_query = sql.SQL(
            """
            ALTER TABLE {topic}.{platform}
            ADD COLUMN IF NOT EXISTS clean_text TEXT
            """
        ).format(
            topic=sql.Identifier(self.topic), platform=sql.Identifier(self.platform)
        )

        # Add topic column
        add_topic_query = sql.SQL(
            """
            ALTER TABLE {topic}.{platform}
            ADD COLUMN IF NOT EXISTS topic TEXT
            """
        ).format(
            topic=sql.Identifier(self.topic), platform=sql.Identifier(self.platform)
        )

        self.db.cursor.execute(add_clean_text_query)
        self.db.cursor.execute(add_topic_query)
        self.db.connection.commit()
        logger.info(
            f"Added clean_text and topic columns to {self.topic}.{self.platform}"
        )

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx : min(ndx + n, l)]

    def detect_language(self, texts):
        results = self.lang_detect(texts)
        langs = [result["label"] for result in results]
        return langs

    def get_chunks(self, data, chunksize):
        """Yield successive n-sized chunks from the dataframe."""
        for i in range(0, len(data), chunksize):
            yield data[i : i + chunksize]

    def update_lang(self, chunksize=1000):
        # Update clean_text and topic columns
        chunks = self.get_chunks(self.data, chunksize)
        total_chunks = len(self.data) // chunksize
        for chunk_index, chunk in enumerate(chunks):
            try:
                update_query = sql.SQL(
                    """
                    UPDATE {topic}.{platform}
                    SET lang = %s
                    WHERE id = %s
                    """
                ).format(
                    topic=sql.Identifier(self.topic),
                    platform=sql.Identifier(self.platform),
                )

                values_list = [(row["lang"], row["id"]) for _, row in chunk.iterrows()]
                self.db.cursor.executemany(update_query, values_list)
                self.db.connection.commit()
                logger.info(
                    f"{chunk_index+1}/{total_chunks} - Updated {len(chunk)} rows in {self.topic}.{self.platform} table"
                )
            except Exception as e:
                first_row_id = chunk.iloc[0]["id"]
                logger.error(
                    f"Error inserting batch {chunk_index + 1} starting with row id {first_row_id}: {e}"
                )
        logger.info("Update lang column in {self.topic}.{self.platform} table")

    def update_clean_text(self, chunksize=1000):
        # Update clean_text and topic columns
        chunks = self.get_chunks(self.data, chunksize)
        total_chunks = len(self.data) // chunksize
        for chunk_index, chunk in enumerate(chunks):
            try:
                update_query = sql.SQL(
                    """
                    UPDATE {topic}.{platform}
                    SET clean_text = %s
                    WHERE id = %s
                    """
                ).format(
                    topic=sql.Identifier(self.topic),
                    platform=sql.Identifier(self.platform),
                )

                values_list = [
                    (row["clean_text"], row["id"]) for _, row in chunk.iterrows()
                ]
                self.db.cursor.executemany(update_query, values_list)
                self.db.connection.commit()
                logger.info(
                    f"{chunk_index+1}/{total_chunks} - Updated {len(chunk)} rows in {self.topic}.{self.platform} table"
                )
            except Exception as e:
                first_row_id = chunk.iloc[0]["id"]
                logger.error(
                    f"Error inserting batch {chunk_index + 1} starting with row id {first_row_id}: {e}"
                )

        logger.info("Update clean_text column in {self.topic}.{self.platform} table")

    def save_to_db(self, chunksize=1000, batch_num=1):
        if self.platform == "twitter":
            topics_stats_name = f"twitter_batch{batch_num}"
        else:
            topics_stats_name = self.platform

        # Add topics info to a new table
        create_table_query = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {schema}.{table} (
                topic INTEGER PRIMARY KEY,
                count INTEGER,
                topic_label TEXT
            )
            """
        ).format(
            schema=sql.Identifier(self.topic),
            table=sql.Identifier(f"{topics_stats_name}_topic_info"),
        )

        # Insert the results into the database
        insert_query = sql.SQL(
            """
            INSERT INTO {schema}.{table} (topic, count, topic_label)
            VALUES (%s, %s, %s)
            ON CONFLICT (topic) DO NOTHING
            """
        ).format(
            schema=sql.Identifier(self.topic),
            table=sql.Identifier(f"{topics_stats_name}_topic_info"),
        )

        topic_info_values = self.topics_df.values.tolist()
        self.db.cursor.execute(create_table_query)
        self.db.cursor.executemany(insert_query, topic_info_values)
        self.db.connection.commit()

        # Update clean_text and topic columns
        logger.info(
            f"Update {len(self.data)} rows in {self.topic}.{self.platform} table"
        )
        chunks = self.get_chunks(self.data, chunksize)
        total_chunks = len(self.data) // chunksize
        for chunk_index, chunk in enumerate(chunks):
            try:
                update_query = sql.SQL(
                    """
                    UPDATE {topic}.{platform}
                    SET topic = %s, batch = %s
                    WHERE id = %s
                    """
                ).format(
                    topic=sql.Identifier(self.topic),
                    platform=sql.Identifier(self.platform),
                )

                values_list = [
                    (row["topic"], row["batch"], row["id"])
                    for _, row in chunk.iterrows()
                ]
                self.db.cursor.executemany(update_query, values_list)
                self.db.connection.commit()
                logger.info(
                    f"{chunk_index+1}/{total_chunks} - Updated {len(chunk)} rows in {self.topic}.{self.platform} table"
                )
            except Exception as e:
                first_row_id = chunk.iloc[0]["id"]
                logger.error(
                    f"Error inserting batch {chunk_index + 1} starting with row id {first_row_id}: {e}"
                )
        logger.info("Update topic column in {self.topic}.{self.platform} table")
