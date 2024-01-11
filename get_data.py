from google.cloud import bigquery
import pandas as pd
from tqdm import tqdm
import click


@click.command()
@click.option(
    "--tldr",
    default=False,
    is_flag=True,
    help="Filter in only selftexts that contain a tldr",
)
def main(tldr: str):
    # Create a client
    client = bigquery.Client()

    # Write your query
    subreddits = "'AmItheAsshole','entitledparents','relationships','JUSTNOMIL','askwomenadvice','Advice','TwoXChromosomes','tifu','unpopularopinion','povertyfinance', 'AskMenAdvice', 'AskMen', 'AskWomen'"
    tables = [
        "2015_12",
        "2016_01",
        "2016_02",
        "2016_03",
        "2016_04",
        "2016_05",
        "2016_06",
        "2016_07",
        "2016_08",
        "2016_09",
        "2016_10",
        "2016_11",
        "2016_12",
        "2017_01",
        "2017_02",
        "2017_03",
        "2017_04",
        "2017_05",
        "2017_06",
        "2017_07",
        "2017_08",
        "2017_09",
        "2017_10",
        "2017_11",
        "2017_12",
        "2018_01",
        "2018_02",
        "2018_03",
        "2018_04",
        "2018_05",
        "2018_06",
        "2018_07",
        "2018_08",
        "2018_09",
        "2018_10",
        "2018_11",
        "2018_12",
        "2019_01",
        "2019_02",
        "2019_03",
        "2019_04",
        "2019_05",
        "2019_06",
        "2019_07",
        "2019_08",
    ]
    min_score = 10
    min_text_length = 1500
    max_text_length = 5000
    top_k = 100

    df_list = []

    for table in tqdm(tables):
        tldr_versions = [
            "tl dr",
            "tl;dr",
            "tldr",
            "tl:dr",
            "tl/dr",
            "tl; dr",
            "tl,dr",
            "tl, dr",
            "tl-dr",
            "tl’dr",
            "tl: dr",
            "tl.dr",
            "tl ; dr",
            "tl dr",
            "tldr;dr",
            "tl ;dr",
            "tl\\\\dr",  # backslash hell
            "tl/ dr",
            "tld:dr",
            "tl;;dr",
            "tltl;dr",
            "tl˜dr",
            "tl / dr",
            "tl :dr",
            "tl - dr",
            "tl\\\\\\\\dr",  # double backslash hell
            "tl. dr",
            "tl:;dr",
            "tl|dr",
            "tl;sdr",
            "tll;dr",
            "tl : dr",
            "tld;dr",
        ]
        tldr_block = (
            "AND (\n"
            + " OR \n".join(
                [f'(selftext LIKE "%{x.upper()}%")' for x in tldr_versions]
            )
            + "\n)\n"
        )
        query = f"""
            SELECT subreddit, selftext, score
            FROM (
                SELECT *,
                    ROW_NUMBER() OVER(PARTITION BY subreddit ORDER BY score DESC) as rn
                FROM `fh-bigquery.reddit_posts.{table}`
                WHERE subreddit IN ({subreddits})
                {tldr_block if tldr else ""}
                AND score > {min_score}
                AND not over_18
                AND LENGTH(selftext) > {min_text_length}
                AND LENGTH(selftext) < {max_text_length}
            ) as ranked
            WHERE rn <= {top_k}
        ;
        """

        # Run the query
        query_job = client.query(query)

        # convert query results to dataframe and append
        df_list.append(query_job.to_dataframe())

    # concatenate all dataframes
    df = pd.concat(df_list)
    output_path = (
        "full_data/reddit_tldr_dataset.jsonl"
        if tldr
        else "full_data/reddit_all_dataset.jsonl"
    )
    df.to_json(output_path, orient="records", lines=True)


if __name__ == "__main__":
    main()
