from utils import cached_openai_call


class SummarizationModel:
    def __init__(self):
        pass

    def forward(x):
        # subclass this method
        return x


class GPTSummarizer(SummarizationModel):
    def __init__(self):
        pass

    def forward(self, full_document, model="gpt-4"):
        lm_input = f"You are a summarization assistant, skilled in summarizing long texts into short ones, without changing their style. You do not change the tone of voice or the perspective the text is written from. For example, if the text is written in first person, you keep it in first person. When you receive a piece of text, you respond only with the summary, nothing else. You always shorten the text to around three sentences. Summarize the following text to exactly 3 sentences:\n\n{full_document}\n\nSummary:"
        completion = cached_openai_call(
            lm_input,
            model=model,
        )
        return completion.choices[0].message.content
