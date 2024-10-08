# Work with explicit schemaaa
from typing import List
from datamodels.chatbot import ChatBot
from models.lm_backbone import LmBackboneModel
from models.model_utils import LanguageModelWrapper
from users import person_features
import json


predict_cq_prompt = (
    "Ask a clarifying question that will help you determine the eligibility of user for benefits as efficiently as possible. Look at this JSON to ask additional information: {info}. Only ask about one fact at a time. "
    "However, if you need to ask something about the user's relations for the first time, also ask the relation's name in the same question. "
    "For example, if you want to ask the user if they have a spouse, you should ask \"Do you have a spouse? If yes, what is their name?\""
)

example_1_existing_data = "{\"Andrew\": {\"name\": \"Andrew\", \"has_ssn\": true}, \"Carmen\": {\"name\": \"Carmen\", \"relation\": \"daughter\"}}\n\n"
example_1_updated_data = "{\"Andrew\": {\"name\": \"Andrew\", \"has_ssn\": true}, \"Carmen\": {\"name\": \"Carmen\", \"age\": 13, \"relation\": \"daughter\"}}\n\n"
example_2_existing_data = "{\"Andrew\": {\"name\": \"Andrew\", \"has_ssn\": true}, \"Carmen\": {\"name\": \"Carmen\", \"age\": 13, \"relation\": \"daughter\"}}\n\n"
example_2_updated_data = "{\"Andrew\": {\"name\": \"Andrew\", \"has_ssn\": true}, \"Carmen\": {\"name\": \"Carmen\", \"age\": 13, \"relation\": \"daughter\"}, , \"Stacy\": {\"name\": \"Stacy\", \"age\": 13, \"relation\": \"spouse\"}}\n\n"

update_schema_prompt = (
    "Here is the user schema: {schema}. You have just received new information about the user (and potentially about their household members), use this schema to fill out new information. For example:\n\n"
    "Dialog turn:\n"
    "Chatbot: What is your daughter Carmen's age?\n"
    "User: She is 13 years old.\n\n"
    "Existing data: \n"
    "{example_1_existing_data}"
    "Your output should look like: \n"
    "{example_1_updated_data}"

    "If a person does not exist in the existing data, add a key with the name of the person and their relation to the User. This information will be available whenever a new person is introduced. For example:\n"
    
    "Dialog turn:\n"
    "Chatbot: Do you have a spouse? If yes, what is their name?\n"
    "User: Yes, my wife's name is Stacy.\n\n"
    "Existing data: \n"
    "{example_2_existing_data}"
    "Your output should look like: \n"
    "{example_2_updated_data}"

    "Your output should only contain a JSON parsable string and nothing else.\n\n"

    "Here is your dialog turn:\n"
    "Chatbot: {cq}\n"
    "User: {answer}\n\n"
    "Here is the existing data:\n"
    "{info}\n\n"
    "Your output goes here:\n"
)

benefits_ready_prompt = {
    "role": "system",
    "content": "Is the information sufficient to determine eligibility of all programs? Answer only in one word True or False.",
}

benefits_prediction_prompt = "Predict the programs for which the user is eligible. Return only a boolean array of length {num_programs}, e.g. {example_array}, where the value at index `i` is true iff the user is eligible for program `i`. Only return the array. Do not return anything else in the response. If a user's eligibility is unclear, make your best guess."


# Function to extract field name and type
def extract_field_and_type(features: list[tuple]) -> list[tuple[str, type]]:
    field_and_type = [(field[0], field[1].args[0].__name__) if hasattr(field[1], 'args') else (field[0], type(field[1])) for field in features]
    return field_and_type

# Get the list of field names and types
field_and_type_list = extract_field_and_type(person_features)

# Output the result
for field_type in field_and_type_list:
    print(field_type)

def example_array(n):
    return str([bool(x % 2) for x in range(n)])

class SchemaFillerChatBot(ChatBot):
    """ "Class for chatbots that can fill a user's schema."""

    def __init__(
        self,
        lm_wrapper: LanguageModelWrapper,
        no_of_programs: str,
        eligibility_requirements: str,
        data_only: bool,
    ):
        """
        ChatBot class for keeping the history of user chat and other functions to determine eligbility for benefits
        """
        self.lm_wrapper = lm_wrapper
        self.lm_backbone = LmBackboneModel(self.lm_wrapper)
        self.num_programs = no_of_programs
        self.eligibility_requirements = eligibility_requirements
        self.data_only = data_only
        self.states = []
        
        # Initialize the data to be empty intially
        self.data = {}

    def post_answer(self, history: List[dict]) -> None:
        """
        Update the notebook based on the most recent dialog turn
        """
        # prompt the lm to update the notebook until it returns a valid update
        for i in range(10):
            history = history.copy()
            prompts = [
                {
                    "role": "system",
                    "content": update_schema_prompt.format(
                        schema=str(field_and_type_list),
                        cq=history[-2]["content"],
                        answer=history[-1]["content"],
                        info=json.dumps(self.data),
                        example_1_existing_data=example_1_existing_data,
                        example_1_updated_data=example_1_updated_data,
                        example_2_existing_data=example_2_existing_data,
                        example_2_updated_data=example_2_updated_data,
                    ),
                },
            ]

            # print("prompts --->", prompts)

            lm_output = self.lm_backbone.forward(prompts, num_completions=i + 1)[-1]
            
            try:
                new_data = json.loads(lm_output)
                self.data = new_data
                self.states.append(new_data)
                break
            except:
                print(
                    f"*** WARNING: Invalid JSON found. Attempting again***"
                )
        print(lm_output)

    

    def predict_cq(self, history) -> str:
        """
        Function to generate clarifying question.
        """
        history = history.copy()
        prompts = [
            {
                "role": "system",
                "content": predict_cq_prompt.format(info=self.data),
            }
        ]

        cq = self.lm_backbone.forward(history + prompts)
        return cq

    def predict_benefits_ready(self, history) -> bool:
        """
        Check whether chatbot history has sufficient information to determine eligbility of all benenfits
        """
        history = history.copy()
        
        lm_output = self.lm_backbone.forward(history + [benefits_ready_prompt])
        return str(lm_output)

    def predict_benefits_eligibility(
        self, history, programs
    ) -> List[bool]:
        """
        Predict what all benefits user or its household is eligible for.
        Return a boolean array of length equal to number of benefits.
        """
        history = history.copy()
        if self.data_only:
            prompts = [
                {"role": "system", "content": json.dumps(self.data)},
                {
                    "role": "system",
                    "content": benefits_prediction_prompt.format(
                        num_programs=self.num_programs,
                        example_array=example_array(self.num_programs),
                    ),
                },
                {
                    "role": "system",
                    "content": "Base your prediction on the JSON given above.",
                },
            ]

            lm_output = self.lm_backbone.forward(prompts)
        else:
            prompts = [
                {
                    "role": "system",
                    "content": benefits_prediction_prompt.format(
                        num_programs=self.num_programs,
                        example_array=example_array(self.num_programs),
                    ),
                }
            ]
            lm_output = self.lm_backbone.forward(history + prompts)
        lm_output = self.extract_prediction(lm_output, programs)
        return lm_output
