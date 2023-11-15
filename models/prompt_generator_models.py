from openai import OpenAI


class PromptGeneratorModel:
    def __init__(self):
        pass

    def forward(x):
        # subclass this method
        return x


class GPTPromptGenerator(PromptGeneratorModel):
    def __init__(self):
        self.client = OpenAI()

    def forward(self, x, n, temperature, model="gpt-4"):
        lm_input=f"Memorize the following document and then follow the instructions below:\n\n{x}\n\nInstructions: Generate an interesting question about the document and the speaker. Ideally the question extends to themes beyond the literal facts in the document."
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                # {
                #     "role": "system",
                #     "content": f"You are a study assistant, skilled in creating interesting questions to ask about documents. Your job is to create a challenging, long-form question that can be answered from the document. When you recieve a document, you respond with a questions about the document. Only generate one question. Do not include any other text.",
                # },
                {"role": "user", "content": lm_input},
            ],
            n=n,
            temperature=temperature,
        )
        openai_output = [completion.choices[i].message.content for i in range(n)]

        return openai_output
