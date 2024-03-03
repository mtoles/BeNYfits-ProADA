
cq_prompt = """
We have the following reddit post:
{reddit_post}

The first comment was a question, and OP provided the following response:
{answer}

Provide a sample question that might have evoked this response, given that the commentor had only read the original post when asking. Provide the question in JSON format, as in {{'question': 'The question?'}}"
"""