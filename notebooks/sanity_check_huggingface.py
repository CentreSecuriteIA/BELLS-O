# %%

from dotenv import load_dotenv


# os.environ["XET_SHOW_PROGRESS"] = "false"
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
load_dotenv(override=True)

from bells_o import Usage
from bells_o.datasets import HuggingFaceDataset
from bells_o.supervisors import AutoHuggingFaceSupervisor


# %%
usage = Usage("content_moderation")
dataset = HuggingFaceDataset("lmsys/toxic-chat", usage, version_name="toxicchat0124")

# %%
print(dataset[0])

# %%
model_kwargs = {"device_map": "auto"}
generation_kwargs = {
    "max_new_tokens": 20,
}


supervisor = AutoHuggingFaceSupervisor.load(
    "google/shieldgemma-27b",
    model_kwargs=model_kwargs,
    # generation_kwargs=generation_kwargs,
)

# %%
pre_processed = supervisor.pre_process(dataset[1]["model_output"])
output = supervisor.judge(pre_processed)

# %%
print(output)

# %%
supervisor.pre_processing[-1](dataset[1]["model_output"])

# %%
print(supervisor.pre_processing[-1](dataset[1]["model_output"])[1:])
result = supervisor._tokenizer.apply_chat_template(
    supervisor.pre_processing[-1](dataset[1]["model_output"])[1:], tokenize=False, add_generation_prompt=True
)
print(result)

# %%
judgement = supervisor(dataset[1]["model_output"])

# %%
print(judgement)
