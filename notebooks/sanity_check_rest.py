# %%
from dotenv import load_dotenv

from bells_o import Usage
from bells_o.datasets import HuggingFaceDataset
from bells_o.supervisors import AutoRestSupervisor


load_dotenv()


# %%
usage = Usage("content_moderation")
dataset = HuggingFaceDataset("lmsys/toxic-chat", usage, version_name="toxicchat0124")

# %%
print(dataset[0]["model_output"])

# %%
supervisor = AutoRestSupervisor.load(
    "azure-prompt-shield", kwargs={"endpoint": "https://content-safety-gratis.cognitiveservices.azure.com/"}
)

# %%
pre_processed = supervisor.pre_process(dataset[1]["model_output"])

# %%
output = supervisor.judge(pre_processed)
print(output[0]["output_raw"])
print(output[0]["metadata"])


# %%
judgement = supervisor._res_map_fn(output[0]["output_raw"], usage)
print(judgement)

# %%
judgement = supervisor(dataset[1]["model_output"])
print(judgement)

# %%
