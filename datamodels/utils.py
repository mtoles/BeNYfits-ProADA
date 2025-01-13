import pandas as pd

# df = pd.read_csv("dataset/benefits_clean.csv")
# # save to jsonl
# df.to_json("dataset/benefits_clean.jsonl", orient="records", lines=True)

from bs4 import BeautifulSoup
from html2text import html2text
import re
import num2words


def extract_top_level_tr(html):
    """
    Extract substrings found between top-level <tr> and </tr> tags, inclusive.
    Nested <tr></tr> tags within top-level ones are included.

    Parameters:
        html (str): The input HTML string.

    Returns:
        list: A list of substrings for each top-level <tr></tr> block.
    """
    # Regular expression to match top-level <tr></tr>, accounting for nested tags
    pattern = re.compile(r"<tr(?=\s|>|$)(.*?)>(.*?)</tr>", re.DOTALL)

    result = []
    stack = []
    current_block = []
    index = 0

    while index < len(html):
        # Check for opening <tr> tag
        open_match = re.match(r"<tr(?=\s|>|$)", html[index:], re.DOTALL)
        if open_match:
            if not stack:
                # Start a new top-level block
                current_block = []
            stack.append("<tr>")
            current_block.append(html[index : index + open_match.end()])
            index += open_match.end()
            continue

        # Check for closing </tr> tag
        close_match = re.match(r"</tr>", html[index:], re.DOTALL)
        if close_match:
            if stack:
                stack.pop()
            current_block.append(html[index : index + close_match.end()])
            index += close_match.end()

            if not stack:
                # End of a top-level block
                result.append("".join(current_block))
                current_block = []
            continue

        # Append other content within <tr> to the current block
        if stack:
            current_block.append(html[index])

        index += 1

    return result


def split_first_td(html):
    # Regular expression to match the first <td>...</td> block
    match = re.search(r"<td>.*?<\/td>", html, re.DOTALL)
    if match:
        first_td = match.group(0)
        rest = html[: match.start()] + html[match.end() :]
        return first_td, rest
    else:
        return "", html


html_path = "dataset/benefits_clean.html"
with open(html_path, "r") as f:
    html_content = f.read()

html_rows = extract_top_level_tr(html_content)


# Using html2text for a richer conversion


def camel_case(s: str):
    # use regex to find all substrings that are digits
    digits = re.findall(r"\d+", s)
    # replace all digits with their corresponding number in words
    for d in digits:
        s = s.replace(d, num2words.num2words(d, to="cardinal"))

    out = ""
    words = s.split(" ")
    for i, w in enumerate(words):
        for j, c in enumerate(w):
            if not c.isalpha():
                continue
            else:
                if j == 0:
                    out += c.upper()
                else:
                    out += c

    return out


rt = []
for r in html_rows:
    name, desc = split_first_td(r)
    rt.append((camel_case(html2text(name).strip()), html2text(desc).strip()))

df = pd.DataFrame(rt, columns=["program_name", "plain_language_eligibility"]).iloc[1:]
df.to_csv("dataset/benefits_clean.csv", index=False)
df.to_json("dataset/benefits_clean.jsonl", orient="records", lines=True)
print("\nRich Text:\n", html_rows)
