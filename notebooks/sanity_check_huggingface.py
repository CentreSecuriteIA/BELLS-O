# %%
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
    "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0",
    model_kwargs=model_kwargs,
    generation_kwargs=generation_kwargs,
)

# %%
pre_processed = supervisor.pre_process(dataset[1]["model_output"])
output = supervisor.judge(pre_processed)

# %%
supervisor._model

# %%
print(output)

# %%
judgement = supervisor(dataset[1]["model_output"])

# %%
print(judgement)
