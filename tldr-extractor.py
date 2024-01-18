import json
import re
import markdown2

jsonl_file_path = "./reddit_scrape/reddit_tldr_dataset.jsonl"
output_jsonl_file = "./reddit_scrape/reddit_tldr_dataset_filtered.jsonl"

# jsonl_file_path = "./reddit_scrape/full_reddit_dataset.jsonl"
# output_jsonl_file = "./reddit_scrape/full_reddit_dataset_filtered.jsonl"

filtered_rows = []

result = []

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

with open(jsonl_file_path, 'r') as file:
    for line in file:    
        data = json.loads(line)

        selftext = data.get('selftext', '')
        subreddit = data.get('subreddit', '')
        score = data.get('score', '')

        # Replace more than two contiguous '\n' with two '\n\n'
        selftext = re.sub(r'\n{3,}', r'\n\n', selftext)

        tldr_indexes = []
        final_tldr_versions = []

        for version in tldr_versions:
            if version in selftext.lower():                
                cur_tldr_index = selftext.lower().find(version)

                tldr_indexes.append(cur_tldr_index)
                final_tldr_versions.append(version)

        if len(tldr_indexes) == 1:
            if tldr_indexes[0] == 0:
                # selftext starting with tldr. split by `\n\n` - Content \n\n Summary
                newlines_index = selftext.find('\n\n')

                if newlines_index != -1:
                    content = selftext[:newlines_index]
                    summary = selftext[newlines_index + len('\n\n'):]

                    summary = summary[1:] if summary.startswith(":") else summary

                    result.append({
                        'subreddit': subreddit,
                        'content': content.strip(),
                        'summary': summary.strip(),
                        'score': score
                    })
            else:
                tldr_index = tldr_indexes[0]
                final_tldr_version = final_tldr_versions[0]

                content = selftext[:tldr_index]
                summary = selftext[tldr_index + len(final_tldr_version):]

                # Remove : from summary if it starts with :
                summary = summary[1:] if summary.startswith(":") else summary

                result.append({
                    'subreddit': subreddit,
                    'content': content.strip(),
                    'summary': summary.strip(),
                    'score': score
                })

with open(output_jsonl_file, 'w') as outfile:
    for row in result:
         outfile.write(json.dumps(row) + '\n')