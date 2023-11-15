from openai import OpenAI


class SummarizationModel:
    def __init__(self):
        pass

    def forward(x):
        # subclass this method
        return x


class GPTSummarizer(SummarizationModel):
    def __init__(self):
        self.client = OpenAI()

    def forward(self, x, model="gpt-4"):
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a summarization assistant, skilled in summarizing long texts into short ones, without changing their style. You do not change the tone of voice or the perspective the text is written from. For example, if the text is written in first person, you keep it in first person. When you receive a piece of text, you respond only with the summary, nothing else. You always shorten the text to around three sentences.",
                },
                {"role": "user", "content": x},
            ],
        )
        return completion.choices[0].message.content
