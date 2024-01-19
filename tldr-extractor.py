import json
import re

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
        # print(summary.strip())

    # The summary length is less than 10% of length of total post then only consider the same
    if len(summary) > 0.10 * len(selftext):
        return
    
    result.append({
        'subreddit': subreddit,
        'doc_orig': content.strip(),
        'doc_summ': summary.strip(),
        'score': score
    })

with open(jsonl_file_path, 'r') as file:
    for line in file:    
        data = json.loads(line)

        selftext = data.get('selftext', '')
        # selftext = "Oh my word. [This](https:\/\/www.reddit.com\/r\/JUSTNOMIL\/comments\/3urf0n\/christmas_drama_already\/) was the original post. TL;DR, mom didn't want to share Christmas, wanted us to drive 4 hours.\n\nSO, mom decided that because she and dad adopted a puppy, she wanted to bring it and just put it in a kennel at the house for the whole time they were here. My niece is terrified of dogs (and she's only about a year and a half old), and my sister's fianc\u00e9 is allergic. So my sister said point blank \"No. If the dog is going to stay in the kennel that long, leave it at your house. I don't want my kiddo screaming through the holidays\". Mom consulted with dad, they both said never mind, enjoy your holidays without us. She proceeded to get on Facebook posting pictures of her grandchildren and whining about how much she missed them and hated this holiday season. Bitch, please. You could have boarded the dog, asked us to meet halfway for dinner, ANYTHING. Then dad calls on Christmas, complains that we only texted him (FYI they only texted us to say Merry Christmas). I had laryngitis over the holidays, and almost no voice. I listened while my dad said my sister \"shut them down\" from visiting, how he they weren't just going to leave their dog at home because \"the dog is like family\". My sister unleashed. And I was so proud of her. She (very respectfully, which amazed me) told them they had other options, they shut themselves down, and her daughter (and their TWO pregnant daughters) should hold more importance then their dog during the holidays. She followed up by saying she refused to be guilt tripped for acting in her family's best interest. She shut my dad DOWN. So proud. And we proceeded to eat cupcakes and give no fucks. WIN FOR US GIRLS!"
        subreddit = data.get('subreddit', '')
        score = data.get('score', '')

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
                    additional_constraint_and_parsing(content, summary)

            else:
                content = selftext[:tldr_index]
                summary = selftext[tldr_index + len(final_tldr_version):]

                additional_constraint_and_parsing(content, summary)    
        # break            

with open(output_jsonl_file, 'w') as outfile:
    for row in result:
         outfile.write(json.dumps(row) + '\n')