from tqdm import tqdm
from typing import List, Optional
from tqdm import tqdm
import os
from models.lm_logging import LmLogger
from inspect import currentframe
from server.model_client import ModelAPIClient, ForwardRequest

tqdm.pandas()
INPUT_TOKEN_LIMIT = 4096
OUTPUT_TOKEN_LIMIT = 4096


class LmBackboneModel:
    def __init__(
        self,
        id_of_model: str,
        use_cache: bool,
        # mode: PromptMode = PromptMode.DEFAULT,
        lm_logger: Optional[LmLogger] = None,
    ):
        # self.lm_wrapper = lm_wrapper
        # self.mode = mode
        self.id_of_model = id_of_model
        self.lm_logger = lm_logger
        self.use_cache = use_cache
        self.client = ModelAPIClient("http://localhost:8000", self.lm_logger)

    def forward(
        self,
        history: List[dict],
        num_completions: Optional[int] = None,
        logging_role: str = "No_Role",
    ) -> str | List[str]:
        self.lm_logger.current_input = history

        forward_request = ForwardRequest(
            name_of_model=self.id_of_model,
            history=history,
            use_cache=self.use_cache,
            constraints=None,
            random_seed=self.client.random_seed,
        )
        output = self.client.forward(forward_request)

        if self.lm_logger:
            first_output = output[-1] if isinstance(output, list) else output
            self.lm_logger.log_io(
                lm_input=history, lm_output=first_output, role=logging_role
            )
        return output


# if __name__ == "__main__":
# ### testing

# chat_history = [
#     {
#         "role": "system",
#         "content": "respond to the user's next question",
#     },
#     {"role": "user", "content": "what is your name?"},
# ]
# gpt_wrapper = load_lm("gpt-3-5-turbo-0125")
# model = LmBackboneModel(gpt_wrapper)
# output = model.forward(chat_history, currentframe())
# print(output)

# llama3_wrapper = load_lm("meta-llama/Meta-Llama-3-8B-Instruct")
# model = LmBackboneModel(llama3_wrapper)
# output = model.forward(chat_history, currentframe())
# print(output)

# # apply model
