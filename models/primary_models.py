from utils import cached_openai_call

class PrimaryModel:
    def __init__(self):
        pass

    def forward(x):
        # subclass this method
        return x


class GPTPrimaryModel(PrimaryModel):
    def __init__(self):
        pass

    def forward(self, document: str, prompt: str, temperature: float, model="gpt-4"):
        lm_input = f"Memorize the following document and then follow the instructions below:\n\n{document}\n\nInstructions: {prompt}"
        completion = cached_openai_call(
            lm_input,
            model=model,
            n=1,
            temperature=temperature,
        )
        openai_output = completion.choices[0].message.content

        return openai_output
