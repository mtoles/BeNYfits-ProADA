from openai import OpenAI


class PrimaryModel:
    def __init__(self):
        pass

    def forward(x):
        # subclass this method
        return x


class GPTPrimaryModel(PrimaryModel):
    def __init__(self):
        self.client = OpenAI()

    def forward(self, document, prompt, temperature, model="gpt-4"):
        lm_input=f"Memorize the following document and then follow the instructions below:\n\n{document}\n\nInstructions: {prompt}"
        completion = self.client.chat.completions.create(
            model=model,
            messages=[

                {"role": "user", "content": lm_input},
            ],
            n=1,
            temperature=temperature,
        )
        openai_output = completion.choices[0].message.content

        return openai_output
