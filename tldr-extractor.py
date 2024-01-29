import json
import re
import pandas as pd
from utils import *

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

doc_orig_col_name = 'doc_orig'
doc_summ_col_name = 'doc_summ'

def parse_and_apply_constraints(selftext, content, summary, df_row):
    """Parse content and summary and apply additional constraints.
    Adds the modified content and summary to the dataframe row and returns the updated dataframe row if applicable."""

    # Summary does not starts with tldr (Stripped off before calling this function)
    # Remove these words from summary if it starts with :;,.
    summary = summary.lstrip(":;,.")
    # summary = summary[1:] if summary.startswith(":" or ";" and so on) else summary

    # If post is of the format - [Content-Part1(tldr)Summary \n\n Content-Part2] i.e. contains new line in summary
    # Then we are concatenating content part1 and part2 to form complete content and portion 
    newlines_index_in_summary = summary.find('\n\n')

    if newlines_index_in_summary != -1:
        content += summary[newlines_index_in_summary:]
        summary = summary[:newlines_index_in_summary]

    # The summary length is less than 10% of length of total post then only consider the same
    if len(summary) > 0.10 * len(selftext):
        return None
    
    df_row[doc_orig_col_name] = content.strip()
    df_row[doc_summ_col_name] = summary.strip()

    return df_row

def split_tldr_posts_into_content_and_summary(df: pd.DataFrame):
    """Function to split the given post into 2 parts - Original Post and Summary applying different criterias"""

    # Regex check to replace more than one contiguous instances of '\n\n' with single '\n\n'
    # Eg: '\n\n\n\n' => '\n\n'
    df['selftext'] = df['selftext'].apply(lambda x: re.sub(r'\n{3,}', r'\n\n', x))
    
    filtered_rows = []

    for _, row in df.iterrows():
        selftext = row['selftext']

        tldr_indexes = []
        final_tldr_versions = []

        for version in tldr_versions:
            if version in selftext.lower():                
                cur_tldr_index = selftext.lower().find(version)

                # Continue searching in selftext until no more occurrences of current version are found
                # Loop for scenarios to look for all instances of current tldr version in the initial post
                while cur_tldr_index != -1:  
                    tldr_indexes.append(cur_tldr_index)
                    final_tldr_versions.append(version)
                    cur_tldr_index = selftext.lower().find(version, cur_tldr_index + 1)
    
        if len(tldr_indexes) != 1:
            continue
        
        # Condition when there is only one instance of tldr present in the text, then only we process
        # Rest all cases are filtered out
        
        tldr_index = tldr_indexes[0]
        final_tldr_version = final_tldr_versions[0]
            
        if tldr_index == 0:
            # selftext starting with tldr. split by first `\n\n` (only if present) - [(tldr)Summary \n\n Content]
            # Discard data points when post starts with tldr but there is no new line in the entire post
            newlines_index = selftext.find('\n\n')

            if newlines_index != -1:
                content = selftext[newlines_index + len('\n\n'):]
                summary = selftext[len(final_tldr_version):newlines_index]
                output_row = parse_and_apply_constraints(selftext, content, summary, row.copy())

                if output_row is not None:
                    filtered_rows.append(output_row.to_frame().T)

        else:
            # selftext does not starts with tldr. Then split by tldr: [Content (tldr)Summary]
            content = selftext[:tldr_index]
            summary = selftext[tldr_index + len(final_tldr_version):]
            output_row = parse_and_apply_constraints(selftext, content, summary, row.copy())
                
            if output_row is not None:
                filtered_rows.append(output_row.to_frame().T)

    output_df = pd.DataFrame(columns=df.columns)
    if filtered_rows:
        output_df = pd.concat(filtered_rows, ignore_index=True)

    return output_df

if __name__ == "__main__":
    input_file_path = "./data/reddit_tldr_dataset.jsonl"
    output_file_path = "./data/reddit_tldr_dataset_filtered_1.jsonl"

    with open(input_file_path, 'r') as file:
        input_data = [json.loads(line) for line in file]

    input_df = pd.DataFrame(input_data)

    output_df = split_tldr_posts_into_content_and_summary(input_df)

    output_df.to_json(output_file_path, orient='records', lines=True)
    df_to_md(output_df, "./results/output_tldr_refined.md")