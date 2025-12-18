from os import getenv

from dotenv import load_dotenv


if load_dotenv(override=True):
    print("loaded variables")
print(getenv("HF_HOME"))

from bells_o.supervisors import AutoHuggingFaceSupervisor


if __name__ == "__main__":
    model_kwargs = {"device_map": "cpu"}
    generation_kwargs = {}

    print("loading model now")
    supervisor = AutoHuggingFaceSupervisor.load(
        "google/shieldgemma-2b",
        model_kwargs=model_kwargs,
        generation_kwargs=generation_kwargs,
    )
