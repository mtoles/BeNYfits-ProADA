from utils import *


class SummarizationModel:
    def __init__(self):
        pass

    def forward(x):
        # subclass this method
        return x


class GPTSummarizer(SummarizationModel):
    def __init__(self, use_cache):
        self.use_cache = use_cache

    def forward(self, full_document, model="gpt-4"):
        lm_input = f"You are a summarization assistant, skilled in summarizing long texts into short ones, without changing their style. You do not change the tone of voice or the perspective the text is written from. For example, if the text is written in first person, you keep it in first person. When you receive a piece of text, you respond only with the summary, nothing else. You always shorten the text to a single sentence. Summarize the following text to exactly one sentence:\n\n{full_document}\n\nSummary:"
        completion = conditional_openai_call(
            lm_input,
            model=model,
            use_cache=self.use_cache
        )
        return completion.choices[0].message.content

class BARTSummarizer(SummarizationModel):
    """BART Summarizer class. Uses the BART Large CNN model to summarize input text.
    """
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    def forward(self, full_document, min_length=30, max_length=130):
        """Forward function of the BART summarizer.

        Args:
            full_document (list[str] or str): A string instance or a list of strings to be summarized.
            min_length (int, optional): The minimum number of tokens in the output. Defaults to 30.
            max_length (int, optional): The maximum number of tokens in the output. Defaults to 130.

        Returns:
            _type_: A list of summaries, if the input is a list. 
            A string with the summary, if the input is str.
        """

        summaries = self.summarizer(full_document, max_length=max_length, min_length=min_length, do_sample=False)

        if isinstance(full_document, list):
            return [summary_record['summary_text'] for summary_record in summaries]
        else:
            return summaries[0]['summary_text']


