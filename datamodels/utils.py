from bs4 import BeautifulSoup

# Read the HTML content from a file
with open("../dataset/benefits_short_v0.0.1.html", "r", encoding="utf-8") as html_file:
    html_content = html_file.read()

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, "html.parser")

# Extract the text from the parsed HTML
parsed_text = soup.get_text()

# Write the extracted text to a text file
with open("../dataset/benefits_short_v0.0.1.txt", "w", encoding="utf-8") as text_file:
    text_file.write(parsed_text)