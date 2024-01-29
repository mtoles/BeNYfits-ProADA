from google.cloud import bigquery
import pandas as pd
from tqdm import tqdm
import click
import json
import re

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
    "tld;dr"
]


def filter_df(df: pd.DataFrame) -> bool:
    filtered_rows = []

    result = []

    def additional_constraint_and_parsing(content, summary):
        # Summary does not starts with tldr (Stripped off before calling this function)
        # Remove these words from summary if it starts with :;,.
        summary = summary.lstrip(":;,.")
        # summary = summary[1:] if summary.startswith(":") else summary

        # If post is of the format - [Content-Part1(tldr)Summary \n\n Content-Part2] i.e. contains new line in summary
        # Then we are concatenating content part1 and part2 to form complete content and portion 
        newlines_index_in_summary = summary.find('\n\n')

        if newlines_index_in_summary != -1:
            content_second_part = summary[newlines_index_in_summary:]
            summary = summary[:newlines_index_in_summary]
            content = content + content_second_part        

        # The summary length is less than 10% of length of total post then only consider the same
        if len(summary) > 0.10 * len(selftext):
            return None
        
        return result

    # with open(jsonl_file_path, 'r') as file:
    for index, row in tqdm(df.iterrows()):
        # for line in file:    
        for line in row['selftext'].splitlines():
            # data = json.loads(line)
            selftext = line.get('selftext', '')
            subreddit = line.get('subreddit', '')
            score = line.get('score', '')

            # Replace more than two contiguous '\n' with two '\n\n'
            selftext = re.sub(r'\n{3,}', r'\n\n', selftext)

            tldr_indexes = []
            final_tldr_versions = []

            for version in tldr_versions:
                if version in selftext.lower():                
                    cur_tldr_index = selftext.lower().find(version)

                    # Continue searching in selftext until no more occurrences of current version are found
                    while cur_tldr_index != -1:  
                        tldr_indexes.append(cur_tldr_index)
                        final_tldr_versions.append(version)
                        cur_tldr_index = selftext.lower().find(version, cur_tldr_index + 1)
        
            if len(tldr_indexes) == 1:
                # Condition when there is only one instance of tldr present in the text
                # Rest all cases are filtered out
                tldr_index = tldr_indexes[0]
                final_tldr_version = final_tldr_versions[0]
                
                if tldr_index == 0:
                    # selftext starting with tldr. split by first `\n\n` - [(tldr)Summary \n\n Content]
                    newlines_index = selftext.find('\n\n')

                    if newlines_index != -1:
                        content = selftext[newlines_index + len('\n\n'):]
                        summary = selftext[len(final_tldr_version):newlines_index]
                        return additional_constraint_and_parsing(content, summary)

                else:
                    content = selftext[:tldr_index]
                    summary = selftext[tldr_index + len(final_tldr_version):]

                    return additional_constraint_and_parsing(content, summary)    

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
        tldr_block = (
            "AND (\n"
            + " OR \n".join(
                [f'(LOWER(selftext) LIKE "%{x.lower()}%")' for x in tldr_versions]
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
                AND LOWER(selftext) LIKE '% i %'
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

    # apply the tldr filter if specified
    if tldr:
        df = filter_df(df)
    output_path = (
        "full_data/reddit_tldr_dataset.jsonl"
        if tldr
        else "full_data/reddit_all_dataset.jsonl"
    )
    df.to_json(output_path, orient="records", lines=True)


if __name__ == "__main__":
    main()
