import os
import argparse

import pandas as pd

# from models.model_utils import load_lm
from datamodels.chatbot import *
from datamodels.humanbot import HumanBot
from datamodels.randombot import RandomBot
from datamodels.syntheticuser import SyntheticUser
from datetime import datetime
from tqdm import tqdm
from acc_over_time_experiment import plot_code_mode_results
from models.lm_logging import LmLogger
from users.dataset_generation import unit_test_dataset
from users.users import Household
from datamodels.codebot import CodeBot
from datetime import datetime
from uuid import uuid4
from users.benefits_programs import BenefitsProgramMeta
from utils import RoleEnum
import json

start = datetime.now()
parser = argparse.ArgumentParser(description="Build benefits bot")
parser.add_argument(
    "--chat_model_id",
    default="meta-llama/Meta-Llama-3-70B-Instruct",
    help="Name of the benefits bot model to use.",
)
parser.add_argument(
    "--code_model_id",
    default=None,
    type=str,
    help="Name of the codellama model to use.",
)
parser.add_argument(
    "--chatbot_strategy",
    default="backbone",
    help="Strategy to use for the benefits bot.",
)
parser.add_argument(
    "--max_code_gen_attempts",
    default=1,
    type=int,
    help="Number of times to attempt to generate code",
)
parser.add_argument(
    "--max_code_rewrite_attempts",
    default=0,
    type=int,
    help="Number of times to attempt to rewrite code",
)
parser.add_argument(
    "--synthetic_user_model_name",
    default="meta-llama/Meta-Llama-3-70B-Instruct",
    help="Name of the synthetic user model to use.",
)
parser.add_argument(
    "--max_dialog_turns",
    default=100,
    help="Maximum number of iterations between benefits bot and synthetic user",
    type=int,
)
parser.add_argument(
    "--eligibility_requirements",
    default="./dataset/benefits_clean.jsonl",
    help="Path to the chat history or benefits description",
)
parser.add_argument(
    "--dataset_path",
    default="dataset/edge_case_dataset.jsonl",
    help="Path to the chat history or benefits description",
)
parser.add_argument(
    "--downsample_size",
    default=None,
    type=int,
    help="Downsample the dataset to this size",
)
parser.add_argument(
    "--ds_shift",
    default=0,
    type=int,
    help="Shift the dataset by this amount",
)
parser.add_argument(
    "--top_k",
    default=20,
    type=int,
    help="Number of similar sentences to pick from natural language profile to match with question in synthetic user",
)

parser.add_argument(
    "--programs",
    default=BenefitsProgramMeta.registry.keys(),
    help="Number of programs in the dataset",
    nargs="+",
)
parser.add_argument(
    "--num_programs",
    default=None,
    type=int,
    help="Downsample to the first n programs",
)
parser.add_argument(
    "--estring",
    default="tmp",
    type=str,
    help="Experiment tracking string",
)
parser.add_argument(
    "--use_cache",
    default=True,
    type=lambda x: (
        (str(x).lower() == "true")
        if str(x).lower() in ("true", "false")
        else (_ for _ in ()).throw(ValueError("Value must be 'true' or 'false'"))
    ),
    help="Use lmwrapper cache. Disable to allow sampling",
)
parser.add_argument(
    "--random_seed",
    default=0,
    type=int,
    help="Random seed to use",
)

TURNS_PER_PROGRAM = 20
args = parser.parse_args()
if args.synthetic_user_model_name == "same":
    args.synthetic_user_model_name = args.chat_model_id
now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
programs_abbreviation = len(args.programs)

output_dir = f"./results/{args.estring}/{now}_{programs_abbreviation}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# print args to the output dir
args_path = os.path.join(output_dir, "params.txt")
args_df = pd.DataFrame(args.__dict__).iloc[:1]
args_df.to_json(args_path)

# Read the chat history from the file
# from server.model_client import gpt_forward_cached
# history = [
#     {
#         "role": "user",
#         "content": "How many words are in the sentence 'Hello World'?",
#     }
# ]
# output = gpt_forward_cached(
#     "gpt-4o-mini-2024-07-18",
#     history,
#     response_format=None,
# )


def read_eligibility_requirements(file_path, num_programs):
    with open(file_path, "r") as file:
        file_content = file.read()
        assert len(file_content) > 0, "File is empty"
        eligibility_df = pd.read_json(file_content, lines=True)
        if num_programs is not None:
            eligibility_df = eligibility_df.iloc[:num_programs]
        return eligibility_df


def eligibility_to_string(eligibility_df):
    eligibility_def = ""
    for index, row in eligibility_df.iterrows():
        eligibility_def += f"**{index}. {row['program']}**\n"
        eligibility_def += f"{row['description']}\n\n"
    return eligibility_def


predictions_df = read_eligibility_requirements(
    args.eligibility_requirements, args.num_programs
)
all_eligibility_requirements = predictions_df.set_index("program_name")[
    "plain_language_eligibility"
].to_dict()
all_eligibility_requirements = {
    k: v for k, v in all_eligibility_requirements.items() if k in args.programs
}

program_names = set(all_eligibility_requirements.keys())
class_names = set(args.programs)
bad_class_names = class_names - program_names
bad_program_names = program_names - class_names
# print(f"Bad class names: {bad_class_names}")
# print(f"Bad program names: {bad_program_names}")
assert len(bad_class_names) == 0

if os.path.exists(args.dataset_path):
    labels_df = pd.read_json(args.dataset_path, lines=True)
elif args.dataset_path == "unittest":
    labels_df = unit_test_dataset()
else:
    raise ValueError(f"Invalid dataset path: {args.dataset_path}")
assert len(labels_df) > 0
labels_df["hh"] = labels_df["hh"].apply(
    lambda hh: Household.from_dict(hh) if isinstance(hh, dict) else hh
)
if args.ds_shift:
    labels_df = labels_df.iloc[args.ds_shift :]
if args.downsample_size:
    labels_df = labels_df[: args.downsample_size]
labels_df.rename(columns={"edge_case_programs": "target_programs"}, inplace=True)
predictions = []
histories = []
per_turn_all_predictions = []
last_turn_iteration = []

num_benefits = len(args.programs)

lm_logger = LmLogger(log_dir=output_dir)


def get_chatbot(
    strategy: str = args.chatbot_strategy,
    no_of_programs: str = len(args.programs),
    eligibility_dict: dict = all_eligibility_requirements,
    use_cache: bool = args.use_cache,
    lm_logger: LmLogger = lm_logger,
    chat_model_id: str = args.chat_model_id,
    code_model_id: Optional[str] = args.code_model_id,
    random_seed: int = args.random_seed,
    data_user_index: int = 0,
    target_programs: Optional[List[str]] = None,
):
    if strategy == "backbone":
        return ChatBot(
            chat_model_id=chat_model_id,
            no_of_programs=no_of_programs,
            eligibility_requirements=eligibility_dict,
            use_cache=use_cache,
            lm_logger=lm_logger,
            random_seed=random_seed,
        )
    elif strategy == "human":
        return HumanBot(
            chat_model_id=chat_model_id,
            no_of_programs=no_of_programs,
            eligibility_requirements=eligibility_dict,
            use_cache=use_cache,
            lm_logger=lm_logger,
            random_seed=random_seed,
            target_programs=target_programs,
        )
    elif strategy == "codebot":
        return CodeBot(
            chat_model_id=chat_model_id,
            no_of_programs=no_of_programs,
            eligibility_requirements=eligibility_dict,
            use_cache=use_cache,
            random_seed=random_seed,
            lm_logger=lm_logger,
            code_model_id=code_model_id,
            max_code_gen_attempts=args.max_code_gen_attempts,
            max_code_rewrite_attempts=args.max_code_rewrite_attempts,
            data_user_index=data_user_index,
        )
    elif strategy == "cot":
        return CotChatBot(
            chat_model_id=chat_model_id,
            no_of_programs=no_of_programs,
            eligibility_requirements=eligibility_dict,
            use_cache=use_cache,
            lm_logger=lm_logger,
            random_seed=random_seed,
        )
    elif strategy == "random":
        return RandomBot(
            chat_model_id=chat_model_id,
            no_of_programs=no_of_programs,
            eligibility_requirements=eligibility_dict,
            use_cache=use_cache,
            lm_logger=lm_logger,
            random_seed=random_seed,
        )
    else:
        raise NotImplementedError(f"Invalid chatbot strategy: {strategy}")


# chatbot = get_chatbot(
#     strategy=args.chatbot_strategy,
#     no_of_programs=len(args.programs),
#     eligibility_dict=all_eligibility_requirements,
#     use_cache=args.use_cache,
#     lm_logger=lm_logger,
#     chat_model_id=args.chat_model_id,
#     code_model_id=args.code_model_id,
# )

generated_code_filename = f"generated_code_{now}_{uuid4()}.py"
generated_code_path = os.path.join("generated_code", generated_code_filename)
os.makedirs("generated_code", exist_ok=True)

generated_code_results = []
for index, row in tqdm(labels_df.iterrows()):
    target_programs = list(set(row["target_programs"]) & set(args.programs))
    n_programs = len(target_programs)
    turn_limit = min(args.max_dialog_turns, n_programs * TURNS_PER_PROGRAM)
    eligibility_requirements = {
        k: v for k, v in all_eligibility_requirements.items() if k in target_programs
    }

    chatbot = get_chatbot(
        strategy=args.chatbot_strategy,
        no_of_programs=len(target_programs),
        eligibility_dict=eligibility_requirements,
        use_cache=args.use_cache,
        lm_logger=lm_logger,
        chat_model_id=args.chat_model_id,
        code_model_id=args.code_model_id,
        data_user_index=index,
        target_programs=target_programs,
    )

    synthetic_user = SyntheticUser(
        row,
        # hh_nl_desc,
        # hh_nl_always_include,
        args.synthetic_user_model_name,
        use_cache=args.use_cache,
        random_seed=args.random_seed,
        lm_logger=lm_logger,
        top_k=args.top_k,
    )
    labels = row[target_programs]
    lm_logger.add_empty_convo(labels.to_dict())

    code_run_mode = "code" in args.chatbot_strategy
    if code_run_mode:
        with open(generated_code_path, "w") as tf:
            chatbot.pre_conversation(
                code_file_handle=tf,
                eligibility_requirements=eligibility_requirements,
                code_model_id=args.code_model_id,
                use_cache=args.use_cache,
            )

        code_results = chatbot.run_generated_code(
            code_file_path=generated_code_path,
            synthetic_user=synthetic_user,
            eligibility_requirements=eligibility_requirements,
            program_names=target_programs,
        )
        generated_code_results.append(code_results)

        # for p_name, p_res in code_results.items():
        #     label = labels[p_name]
        #     # print labela nd pred
        #     print(f"Program: {p_name}")
        #     print(f"Label: {label}")
        #     print(f"Eligibility: {p_res['eligibility']}")
        #     print("\n")

        predictions_log_entry = {}
        for k, v in code_results.items():
            predictions_log_entry[k] = 1 if v["eligibility"] else 0
        lm_logger.log_predictions([predictions_log_entry])

        code_passed = True
        # We skip the "fallback" conversation if code was run
        # but if you wish to fallback on error, you'd add logic here
        non_code_preds_df = pd.DataFrame([predictions_log_entry])
        continue

    # If not code mode or fallback:
    per_turn_predictions = []
    history = [
        # {
        #     "role": RoleEnum.SYSTEM.value,
        #     "content": (
        #         f"You are a language model trying to help user to determine "
        #         f"eligbility of user for benefits. Currently, you do not know "
        #         f"anything about the user. Ask questions that will help you determine "
        #         f"the eligibility of user for benefits as quickly as possible. "
        #         f"Ask only one question at a time. The eligibility requirements "
        #         f"are as follows:\n\n{eligibility_requirements}"
        #     ),
        # }
    ]
    print(f"Index: {index}")

    cur_iter_count = 0
    decision = None

    while True:
        if (
            cur_iter_count > 0
            and str(chatbot.predict_benefits_ready(history)) == "True"
        ) or cur_iter_count == turn_limit:
            # print(f"Benefits eligibility decided on turn {cur_iter_count}/{turn_limit}")
            # decision = per_turn_predictions[-1]
            decision = chatbot.predict_benefits_eligibility(history, target_programs)
            per_turn_predictions.append(decision)
            # print(f"Decision:  {decision}")
            # print(f"label:     {labels.to_dict()}")
            # print("==" * 20)
            per_turn_predictions.extend([decision] * (turn_limit - cur_iter_count))
            last_turn_iteration.append(cur_iter_count)
            break

        cq = chatbot.predict_cq(history, chat_model_id=args.chat_model_id)
        history.append({"role": RoleEnum.CQ_MODEL.value, "content": cq})
        cq_answer = synthetic_user.answer_cq(history=history, cq=cq)
        history.append({"role": RoleEnum.SYNTHETIC_USER.value, "content": cq_answer})

        # print(f"Turn Number:         {cur_iter_count}")
        # print(f"Clarifying Question: {cq}")
        # print(f"Answer:              {cq_answer}")
        chatbot.post_answer(history)
        # print("==" * 20)
        cur_iter_count += 1

    per_turn_all_predictions.append(per_turn_predictions)
    lm_logger.log_predictions(per_turn_predictions)
    lm_logger.log_hh_diff(row["hh"])

lm_logger.save()

turns = []
for log in lm_logger.log:
    count = 0
    dialog = log["dialog"]
    for convo in dialog:
        if convo[-1]["role"] in ["predict_cq", "key_error"]:
            count += 1
    turns.append(count)


if code_run_mode:
    eligibility_li = []
    completed_li = []
    for i, d in enumerate(generated_code_results):
        # d is a dict: { program_name: { 'eligibility': bool, 'completed': bool, ...}, ...}
        eligibility_line = {}
        completed_line = {}
        for pn, dd in d.items():
            eligibility_line[pn] = dd["eligibility"]
            completed_line[pn] = dd["completed"]
        eligibility_li.append(eligibility_line)
        completed_li.append(completed_line)

    eligibility_li_int = [
        {k: (1 if v is True else 0 if v is False else v) for k, v in d.items()}
        for d in eligibility_li
    ]
    predictions_df = pd.DataFrame(eligibility_li_int)
    completed_df = pd.DataFrame(completed_li)

    plot_code_mode_results(
        predictions_df,
        labels_df[args.programs].reset_index(),
        output_dir=output_dir,
        experiment_params={
            "Backbone Model": args.chat_model_id,
            "Strategy": f"{args.estring} {args.chatbot_strategy}",
            "Programs": ", ".join(args.programs),
            "Max Dialog Turns": args.max_dialog_turns,
            "Downsample Size": args.downsample_size,
            "Top K Sentences": args.top_k,
        },
    )
    predictions_df.to_json(
        f"{output_dir}/predictions.jsonl", orient="records", lines=True
    )
    if not completed_df.empty:
        completed_df.to_json(
            f"{output_dir}/completed.jsonl", orient="records", lines=True
        )

else:
    non_code_preds_df = pd.DataFrame([x[-1] for x in per_turn_all_predictions])
    plot_code_mode_results(
        non_code_preds_df,
        labels_df[args.programs].reset_index(),
        output_dir=output_dir,
        experiment_params={
            "Backbone Model": args.chat_model_id,
            "Strategy": f"{args.estring} {args.chatbot_strategy}",
            "Programs": ", ".join(args.programs),
            "Max Dialog Turns": args.max_dialog_turns,
            "Downsample Size": args.downsample_size,
            "Top K Sentences": args.top_k,
        },
    )
    non_code_preds_df.to_json(
        f"{output_dir}/predictions.jsonl", orient="records", lines=True
    )

labels_df[args.programs].astype(int).to_json(
    f"{output_dir}/labels.jsonl", orient="records", lines=True
)

runtime = datetime.now() - start
print(f"Runtime: {runtime}")
print(f"Saved to {output_dir}")

# # print eligibility prediction for each program in args.programs
# for program in args.programs:
#     print(f"{program}: {all_eligibility_requirements[program]}")
#     if program in predictions_df.columns:
#         print(
#             f"Eligibility Prediction: {'Yes' if predictions_df.iloc[0][program] else 'No'}"
#         )
#     else:
#         print(f"Eligibility Prediction: {per_turn_predictions[-1][program]}")
#     print("==" * 20)
