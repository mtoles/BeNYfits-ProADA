from utils import *


class PromptGeneratorModel:
    """
    Model for generating prompts. Subclass this class to implement a new prompt generator model.
    """

    def __init__(self):
        pass

    def forward(document_full: str) -> str:
        # subclass this method
        return document_full


class GPTPromptGenerator(PromptGeneratorModel):
    def __init__(self, use_cache):
        self.use_cache = use_cache

    def forward(self, document_full: str, temperature: float, model="gpt-4") -> str:
        """
        Parameters:
            document_full (str): the full document
            temperature (float): the temperature to use for the GPT model
            model (str): the name of the OpenAI model to use

        Returns:
            str: the generated prompt
        """
        # lm_input = f"Memorize the following document and then follow the instructions below:\n\n{document_full}\n\nInstructions: Generate an interesting question about the document and the speaker. Ideally the question extends to themes beyond the literal facts in the document."
        lm_input = f"Memorize the following document and then follow the instructions below:\n\n{document_full}\n\nInstructions: Generate an interesting question that requires the entire document in order to answer well. Do not ask a multi-part question. Do not ask merely for a summary nor for the sequence of events in the document. Ask a question that requires the reader to think about the document in a new way or provide advice to the speaker."
        completion = conditional_openai_call(
            lm_input,
            use_cache=self.use_cache,
            temperature=temperature,
            model=model,
        )
        openai_output = completion.choices[0].message.content

        return openai_output
