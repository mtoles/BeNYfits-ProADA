from utils import cached_openai_call


class PromptGeneratorModel:
    def __init__(self):
        pass

    def forward(x):
        # subclass this method
        return x


class GPTPromptGenerator(PromptGeneratorModel):
    def __init__(self):
        pass

    def forward(self, document_full: str, n: int, temperature: float, model="gpt-4"):
        lm_input = f"Memorize the following document and then follow the instructions below:\n\n{document_full}\n\nInstructions: Generate an interesting question about the document and the speaker. Ideally the question extends to themes beyond the literal facts in the document."
        completion = cached_openai_call(
            lm_input,
            model=model,
            n=n,
            temperature=temperature,
        )
        openai_output = [completion.choices[i].message.content for i in range(n)]

        return openai_output
