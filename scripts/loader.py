from argparse import ArgumentParser
from os import getenv

from dotenv import load_dotenv


parser = ArgumentParser("loader.py")

parser.add_argument("model_id", nargs="?", help="model_id of model to download")


if load_dotenv(override=True):
    print("loaded variables")
print(getenv("HF_HOME"))

from bells_o.supervisors import AutoHuggingFaceSupervisor


if __name__ == "__main__":
    model_kwargs = {"device_map": "cpu"}
    generation_kwargs = {}

    args = parser.parse_args()

    script_model_id = "ibm-granite/granite-guardian-3.2-5b"

    chosen_model_id = args.model_id or script_model_id
    print(f"loading {chosen_model_id}")

    print("loading model now")
    supervisor = AutoHuggingFaceSupervisor.load(
        chosen_model_id,
        model_kwargs=model_kwargs,
        generation_kwargs=generation_kwargs,
    )
