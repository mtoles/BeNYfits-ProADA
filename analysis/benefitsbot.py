import os
import argparse

import pandas as pd

# from models.model_utils import load_lm
from datamodels.chatbot import *
from datamodels.syntheticuser import SyntheticUser
from datetime import datetime
from tqdm import tqdm
from acc_over_time_experiment import plot_metrics_per_turn, plot_code_mode_results
from models.lm_logging import LmLogger
from users.dataset_generation import unit_test_dataset
from users.users import Household
from datamodels.codebot import CodeBot
from datetime import datetime
from uuid import uuid4
from users.benefits_programs import BenefitsProgramMeta

start = datetime.now()
parser = argparse.ArgumentParser(description="Build benefits bot")
parser.add_argument(
    "--chat_model_id",
    default="meta-llama/Meta-Llama-3-8B-Instruct",
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
    "--synthetic_user_model_name",
    default="meta-llama/Meta-Llama-3-8B-Instruct",
    help="Name of the synthetic user model to use.",
)
parser.add_argument(
    "--max_dialog_turns",
    default=float("inf"),
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

args = parser.parse_args()

now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
programs_abbreviation = len(args.programs)

output_dir = f"./results/{args.estring}/{now}_{programs_abbreviation}"


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
eligibility_requirements = predictions_df.set_index("program_name")[
    "plain_language_eligibility"
].to_dict()
eligibility_requirements = {
    k: v for k, v in eligibility_requirements.items() if k in args.programs
}

program_names = set(eligibility_requirements.keys())
class_names = set(args.programs)
bad_class_names = class_names - program_names
bad_program_names = program_names - class_names
print(f"Bad class names: {bad_class_names}")
print(f"Bad program names: {bad_program_names}")
assert len(bad_class_names) == 0

if os.path.exists(args.dataset_path):
    df = pd.read_json(args.dataset_path, lines=True)
elif args.dataset_path == "unittest":
    df = unit_test_dataset()
df["hh"] = df["hh"].apply(
    lambda hh: Household.from_dict(hh) if isinstance(hh, dict) else hh
)
if args.ds_shift:
    df = df.iloc[args.ds_shift :]
if args.downsample_size:
    df = df[: args.downsample_size]

predictions = []
histories = []
per_turn_all_predictions = []
last_turn_iteration = []

num_benefits = len(args.programs)

lm_logger = LmLogger(log_dir=output_dir)


def get_chatbot(
    strategy: str = args.chatbot_strategy,
    no_of_programs: str = len(args.programs),
    eligibility_dict: dict = eligibility_requirements,
    use_cache: bool = args.use_cache,
    lm_logger: LmLogger = lm_logger,
    chat_model_id: str = args.chat_model_id,
    code_model_id: Optional[str] = args.code_model_id,
):
    if strategy == "backbone":
        return ChatBot(
            chat_model_id=chat_model_id,
            no_of_programs=no_of_programs,
            eligibility_requirements=eligibility_dict,
            use_cache=use_cache,
            lm_logger=lm_logger,
        )
    elif strategy == "codebot":
        return CodeBot(
            chat_model_id=chat_model_id,
            no_of_programs=no_of_programs,
            eligibility_requirements=eligibility_dict,
            use_cache=use_cache,
            lm_logger=lm_logger,
            code_model_id=code_model_id,
        )
    else:
        raise NotImplementedError(f"Invalid chatbot strategy: {strategy}")


chatbot = get_chatbot(
    strategy=args.chatbot_strategy,
    no_of_programs=len(args.programs),
    eligibility_dict=eligibility_requirements,
    use_cache=args.use_cache,
    lm_logger=lm_logger,
    chat_model_id=args.chat_model_id,
    code_model_id=args.code_model_id,
)

generated_code_filename = f"generated_code_{now}_{uuid4()}.py"
generated_code_path = os.path.join("generated_code", generated_code_filename)
os.makedirs("generated_code", exist_ok=True)

generated_code_results = []
for index, row in tqdm(df.iterrows()):
    chatbot = get_chatbot(
        strategy=args.chatbot_strategy,
        no_of_programs=len(args.programs),
        eligibility_dict=eligibility_requirements,
        use_cache=args.use_cache,
        lm_logger=lm_logger,
        chat_model_id=args.chat_model_id,
        code_model_id=args.code_model_id,
    )
    synthetic_user = SyntheticUser(
        row,
        args.synthetic_user_model_name,
        use_cache=args.use_cache,
        lm_logger=lm_logger,
        top_k=args.top_k,
    )
    labels = row[args.programs]
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
            program_names=args.programs,
        )
        generated_code_results.append(code_results)

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
        {
            "role": "system",
            "content": (
                f"You are a language model trying to help user to determine "
                f"eligbility of user for benefits. Currently, you do not know "
                f"anything about the user. Ask questions that will help you determine "
                f"the eligibility of user for benefits as quickly as possible. "
                f"Ask only one question at a time. The eligibility requirements "
                f"are as follows:\n\n{eligibility_requirements}"
            ),
        }
    ]
    print(f"Index: {index}")

    cur_iter_count = 0
    decision = None

    while True:
        if cur_iter_count != 0:
            per_turn_predictions.append(
                chatbot.predict_benefits_eligibility(history, args.programs)
            )
        else:
            default_predictions = dict([(x, 0) for x in args.programs])
            per_turn_predictions.append(default_predictions)

        if cur_iter_count >= args.max_dialog_turns:
            print(f"Max dialog turns ({args.max_dialog_turns}) reached")
            print("==" * 20)
            last_turn_iteration.append(cur_iter_count)
            decision = per_turn_predictions[-1]
            print(f"Decision:  {decision}")
            print(f"label:     {labels.to_dict()}")
            print("==" * 20)
            break

        if (
            cur_iter_count > 0
            and str(chatbot.predict_benefits_ready(history)) == "True"
        ):
            print(f"Benefits eligibility decided on turn {cur_iter_count}/{args.max_dialog_turns}")
            decision = per_turn_predictions[-1]
            print(f"Decision:  {decision}")
            print(f"label:     {labels.to_dict()}")
            print("==" * 20)
            per_turn_predictions.extend(
                [decision] * (args.max_dialog_turns - cur_iter_count)
            )
            last_turn_iteration.append(cur_iter_count)
            break

        cq = chatbot.predict_cq(history, chat_model_id=args.chat_model_id)
        history.append({"role": "assistant", "content": cq})
        cq_answer = synthetic_user.answer_cq(cq)
        history.append({"role": "user", "content": cq_answer})

        print(f"Turn Number:         {cur_iter_count}")
        print(f"Clarifying Question: {cq}")
        print(f"Answer:              {cq_answer}")
        chatbot.post_answer(history)
        print("==" * 20)
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

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
        df[args.programs].reset_index(),
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
        completed_df.to_json(f"{output_dir}/completed.jsonl", orient="records", lines=True)

else:
    non_code_preds_df = pd.DataFrame([x[-1] for x in per_turn_all_predictions])
    plot_code_mode_results(
        non_code_preds_df,
        df[args.programs].reset_index(),
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

df[args.programs].astype(int).to_json(
    f"{output_dir}/labels.jsonl", orient="records", lines=True
)

runtime = datetime.now() - start
print(f"Runtime: {runtime}")
