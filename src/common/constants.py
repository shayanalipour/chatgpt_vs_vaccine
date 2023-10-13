# Use a constant logger name for the whole project
LOGGER_NAME = "vir_conv_logger"


UNIFIED_COLUMNS = {
    "chatgpt": [
        "id TEXT PRIMARY KEY",
        "author_id TEXT NOT NULL",
        "date DATE",
        "text TEXT",
        "lang TEXT DEFAULT 'undefined'",  # Default value set to 'undefined'
        "interaction INTEGER",
    ],
    "covid": [
        "id TEXT PRIMARY KEY",
        "author_id TEXT NOT NULL",
        "date DATE",
        "lang TEXT DEFAULT 'undefined'",
        "interaction INTEGER",
    ],
    "post_url": [
        "id TEXT NOT NULL",
        "url TEXT NOT NULL",
        "tone FLOAT NOT NULL",
        "topic TEXT NOT NULL",
        "platform TEXT NOT NULL",
        "PRIMARY KEY (id, url, platform)",
    ],
    "news": [
        "id TEXT PRIMARY KEY",
        "date DATE NOT NULL",
        "avr_tone FLOAT",
        "polarity FLOAT",
        "domain TEXT NOT NULL",
    ],
    "daily_post_counts": [
        "date DATE NOT NULL",
        "posts INTEGER NOT NULL",
        "cumulative_posts INTEGER NOT NULL",
        "platform VARCHAR(255) NOT NULL",
        "topic VARCHAR(255) NOT NULL",
        "PRIMARY KEY (date, platform, topic)",
    ],
    "interaction_dist": [
        "interaction_count INTEGER NOT NULL",
        "post_count INTEGER NOT NULL",
        "platform VARCHAR(255) NOT NULL",
        "topic VARCHAR(255) NOT NULL",
        "PRIMARY KEY (platform, topic, interaction_count)",
    ],
    "daily_user_counts": [
        "date DATE NOT NULL",
        "cumulative_unique_users INTEGER NOT NULL",
        "platform VARCHAR(255) NOT NULL",
        "topic VARCHAR(255) NOT NULL",
        "PRIMARY KEY (date, platform, topic)",
    ],
}


COLUMN_CONFIG = {
    "instagram": {
        "User Name": "author_id",
        "Post Created Date": "date",
        "Total Interactions": "interaction",
        "URL": "id",
        "Description": "text_1",
        "Image Text": "text_2",
    },
    "reddit": {
        "URL": "id",
        "author_id": "author_id",
        "Post Created Date": "date",
        "Link Text": "text_1",
        "Description": "text_2",
        "Total Interactions": "interaction",
    },
    "youtube": {
        "video_id": "id",
        "channelId": "author_id",
        "publishedAt": "date",
        "title": "text_1",
        "description": "text_2",
        "tags": "text_3",
        # "viewCount": "interaction_1", # Not available in other platforms
        "likeCount": "interaction_2",
        "commentCount": "interaction_3",
    },
    "facebook": {
        "Facebook Id": "author_id",
        "Total Interactions": "interaction",
        "URL": "id",
        "Post Created Date": "date",
        "Message": "text_1",
        "Description": "text_2",
        "Link Text": "text_3",
    },
    "twitter": {
        "id": "id",
        "author_id": "author_id",
        "created_at": "date",
        "text": "text",
        "lang": "lang",
        "retweet_count": "interaction_1",
        "reply_count": "interaction_2",
        "like_count": "interaction_3",
        "quote_count": "interaction_4",
        "retweeted_id": "retweeted_id",
    },
    "news": {
        "DocumentIdentifier": "id",
        "DATE": "date",
        "V2Tone": "tone",
    },
}

POST_URL_COLUMN_CONFIG = {
    "instagram": {
        "URL": "id",
    },
    "reddit": {
        "URL": "id",
        "Link": "target_url",
        "Final Link": "expanded_url",
    },
    "youtube": {
        "video_id": "id",
    },
    "facebook": {
        "URL": "id",
        "Link": "target_url",
        "Final Link": "expanded_url",
    },
    "twitter": {
        "id": "id",
        "unwound_url": "target_url",  # multiple urls separated by space. if "None", then ignore
    },
}


PLATFORM_URL_TABLE_COLUMNS = {
    "instagram": {"id TEXT NOT NULL", "url TEXT"},
    "reddit": {"id TEXT NOT NULL", "target_url TEXT", "expanded_url TEXT"},
    "youtube": {"id TEXT NOT NULL", "url TEXT"},
    "facebook": {
        "id TEXT NOT NULL",
        "target_url TEXT",
        "expanded_url TEXT",
    },
    "twitter": {"id TEXT NOT NULL", "target_url TEXT"},
}


DATE_RANGE = {
    "chatgpt": {"start": "2022-11-25", "end": "2023-02-25"},
    "covid": {"start": "2020-11-01", "end": "2021-02-01"},
}

TOPIC_TABLES = [
    {
        "schema_name": "chatgpt",
        "table_name": [
            "facebook",
            "instagram",
            "news",
            "reddit",
            "twitter",
            "youtube",
        ],
    },
    {
        "schema_name": "covid",
        "table_name": ["facebook", "instagram", "news", "reddit", "twitter", "youtube"],
    },
]


TOPIC_MODELING_PARAMS = {
    "facebook": {
        "umap_neighbors": 30,
        "umap_components": 10,
        "hdbscan_min_cluster_size": 50,
        "hdbscan_min_samples": 20,
        "top_n_words": 5,
    },
    "instagram": {
        "umap_neighbors": 3,
        "umap_components": 20,
        "hdbscan_min_cluster_size": 20,
        "hdbscan_min_samples": 10,
        "top_n_words": 5,
    },
    "reddit": {
        "umap_neighbors": 5,
        "umap_components": 10,
        "hdbscan_min_cluster_size": 20,
        "hdbscan_min_samples": 10,
        "top_n_words": 5,
    },
    "twitter": {
        "umap_neighbors": 150,
        "umap_components": 8,
        "hdbscan_min_cluster_size": 200,
        "hdbscan_min_samples": 100,
        "top_n_words": 5,
    },
    "youtube": {
        "umap_neighbors": 5,
        "umap_components": 20,
        "hdbscan_min_cluster_size": 20,
        "hdbscan_min_samples": 15,
        "top_n_words": 5,
    },
}

VACCINE_KEYWORDS = [
    "no-vax",
    "novax",
    "novaxx",
    "anti-vax",
    "pro-vax",
    "dose",
    "doses",
    "dosers",
    "dosed",
    "pharmaceutical",
    "pharmaceuticals",
    "pharmacies",
    "pharmacist",
    "pharmacists",
    "pharmacology",
    "pharmacotherapy",
    "pharmacy",
    "pharming",
    "pharmings",
    "vaccina",
    "vaccinal",
    "vaccinas",
    "vaccinate",
    "vaccinated",
    "vaccinates",
    "vaccinating",
    "vaccination",
    "vaccinations",
    "vaccinator",
    "vaccinators",
    "vaccine",
    "vaccinee",
    "vaccinees",
    "vaccines",
    "vaccinia",
    "vaccinial",
    "vaccinias",
    "immune",
    "immunes",
    "immunise",
    "immunised",
    "immunises",
    "immunising",
    "immunities",
    "immunity",
    "immunization",
    "immunizations",
    "immunize",
    "immunized",
    "immunizer",
    "immunizers",
    "immunizes",
    "immunizing",
    "immunoassay",
    "immunoassayable",
    "immunoassays",
    "immunoblot",
    "immunoblots",
    "immunoblotting",
    "immunoblottings",
    "immunochemical",
    "immunochemist",
    "immunochemistry",
    "immunochemists",
    "immunocompetent",
    "immunodeficient",
    "immunodiagnoses",
    "immunodiagnosis",
    "immunodiffusion",
    "immunogen",
    "immunogeneses",
    "immunogenesis",
    "immunogenetic",
    "immunogenetics",
    "immunogenic",
    "immunogenicity",
    "immunogens",
    "immunoglobulin",
    "immunoglobulins",
    "immunologic",
    "immunological",
    "immunologically",
    "immunologies",
    "immunologist",
    "immunologists",
    "immunology",
    "immunomodulator",
    "immunopathology",
    "immunoreactive",
    "immunosorbent",
    "immunosorbents",
    "immunosuppress",
    "immunotherapies",
    "immunotherapy",
]