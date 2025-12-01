# BELLS Operational (BELLS-O)
This library aims to bridge benchmarking and AI supervision systems for content-moderation, jailbreaking, and prompt injection. We provide a comprehensive framework to evaluate supervision systems across cost, latency and accuracy.

## Installation
We recommend using uv:
```
uv add git+https://github.com/CentreSecuriteIA/BELLS-O.git
uv sync
```

Alternatively, install the package using pip directly from the repository:
```
pip install git+https://github.com/CentreSecuriteIA/BELLS-O.git
```

or by cloning the repository manually:
```
git clone https://github.com/CentreSecuriteIA/BELLS-O.git
pip install -e BELLS-O
```

## Usage Guide
### The Idea
The framework classifies supervision systems to be a member of one or more of the following categories:
- Content Moderation
- Jailbreak safe guards
- Prompt Injection safe guards

Different supervision systems are wrapped by a `Supervisor` class to enable a unified interface for easy benchmarking, configuration, comparison, etc.

### Quickstart
To run a prompt on the XGuard supervisor by SAIL on the BELLS-O dataset, start your script as follows:
```
from bells_o.datasets import HuggingFaceDataset
from bells_o.supervisors.huggingface import AutoHuggingFaceSupervisor
from bells_o import Usage

usage = Usage("content_moderation") # BELLS-O is a content moderation dataset

dataset = HuggingFaceDataset("bellsop/BELLS-O_Dataset", usage)

model_kwargs = {"device_map": "auto"}
supervisor = AutoHuggingFaceSupervisor("saillab/x-guard", model_kwargs=model_kwargs)

result = supervisor(datset["prompt"][0])
```

For more elaborate runs, consider using the `Evaluator` class.

### Contributing implementations for supervision systems
#### Preliminaries
TODO: Implement pre-commit hooks for ruff
Please enable pre-commit hooks for this repository to adhere to our formatting.

Supervision systems consist of 4 things:
1. The base class of the implementation (e.g. REST or HuggingFace)
2. A function that maps the prompt to the necessary input format.
3. A ResultMapper function that maps the output of the system to a Result object
4. Auxiliary functions or parameters that are specific to each individual implementation

#### Which base class to use?
If you are implementing a supervisor from HuggingFace, choose the `HuggingFaceSupervisor` as a base class, if it is a REST endpoint, choose the `RestSupervisor` as a base class.

#### What is a ResultMapper function?
A `ResultMapper` function is a callable that takes a string (in the case of HF supervisors) or a dictionary (json output in the case of REST supervisors), and a `Usage` object, and outputs a `Result` object.
The goal is to parse the string or dictionary for the necessary information to set the values of the usage types specified in the `Usage` object to the corresponding boolean in the `Result` object. Note that the `usage` argument must always be specified, but for most implementations, it may be ignored since this result mapper is for a specific model, which only supports one usage type anyway.
These functions also have to take care of multi-category flagging and float-to-bool conversions. As a rule, since all harmful prompts should be clearly harmful, we set the standard float threshold for flagging to 0.85 if not otherwise specified by the supervisor documentation. If the documentation provides threshold guidelines, then these take priority over our standard threshold. If there are multiple categories that are analysed for, only one has to trigger to make the prompt harmful.
All `ResultMapper` functions are stored in `bells_o/result_mappers/` in a file that describes the supervisor model. No sub modules. The `__init__.py` imports all mapper functions directly from the files.

#### Implementation for HuggingFace supervisors
##### Step 1: The correct module structures
For HuggingFace models, the modules are structured as: `bells_o.supervisors.huggingface.<lab_name>.<model_name>`. Each lab module directly imports the supervisor classes in its `__init__.py`, and the h`uggingface/__init__.py` directly imports the supervisor classes from all submodules.

##### Step 2: The correct attributes and `__init__` function
Every concrete Implementation is defined by its `__init__` function, which at least has to take the arguments `pre_processing`, `model_kwargs`, `tokenizer_kwargs`, `generation_kwargs`. This function has to set at least the attributes:
- `self.name`: the model id from HuggingFace. It is used to load the model and so has to be the exact string.
- `self.usage`: a Usage object that defines the usage type of the supervisor
- `self.res_map_fn`: the result mapper function for this supervisor.
- Forward passes for the arguments of the `__init__`: `self.pre_processing`, `self.model_kwargs`, `self.tokenizer_kwargs`, `self.generation_kwargs`

##### Step 3: The function that maps the prompt to the input format
For most HF models, the `RoleWrapper` is enough to make all necessary adjustments. Check out its doc string in `preprocessors/role_wrapper.py.` This preprocessor then should be appended to the `pre_processing` list passed in the `__init__` function before setting the attribute. For a more complex set up, check out the role wrapper set up in `saillab/xguard_supervisor.py`.

##### Step 4: The `ResultMapper` function
Since the decoded output will be a string, most result mappers will regex-parse this string to find a certain flag and then extract the value of this flag. However, this is different for every single model and could be virtually of any format. Be ready to be surprised (or something similar with a less positive connotation).

##### Step 5: Auxiliary stuff
For some supervisors, you might need to add extra arguments to the `__init__` function, or get creative in other ways because they require special (non-pre-determined) input formats, etc. For example, check out `openai/gpt_oss_supervisor.py`.

##### Step 6: Add to the `AutoHuggingFaceSupervisor` mappings
At the top of the `supervisors/huggingface/auto_model.py` there is a mapping dict called `MAPPING_DICT`. This dictionary maps the model id string that is used to load the model from HuggingFace to the submodule name in which it is implemented, the class name of its implementation, as well as a dict with initialization keyword arguments. Make sure to add the newly implemented supervisor to this dictionary.

#### Implementation for REST supervisors
##### Step 1: The correct module structures
For HuggingFace models, the modules are structured as: `bells_o.supervisors.rest.<provider_name>.<endpoint_name>`. Each provider module directly imports the supervisor classes in its `__init__.py`, and the `rest/__init__.py` directly imports the supervisor classes from all submodules.

##### Step 2: The correct attributes and `__init__` function
Every concrete Implementation is defined by its `__init__` function, which at least has to take the arguments` pre_processing`, `api_key`, `api_variable`. This function has to set at least the attributes:
- `self.name`: the name of the supervisor.
- `self.provider_name`: the name of the provider of the supervisor.
- `self.base_url`: the REST base URL.
- `self.usage`: a Usage object that defines the usage type of the supervisor
- `self.res_map_fn`: the result mapper function for this supervisor.
- `self.req_map_fn`: a RequestMapper function that takes care of fitting the prompt in the body payload for a POST request.
- `self.auth_map_fn`: an AuthMapper function that creates a header payload for authentication.
- Forward passes for the arguments of the `__init__`: `self.pre_processing`, `self.api_key`, `self.api_variable`
- Optionally, `self.custom_header` can be set, because some endpoints need more information than just the authentication headers. The authentication and custom headers are combined before passing them into the POST request.

##### Step 3: The RequestMapper function
REST APIs are non-standardised and different models have different arguments, and so every model has their own request format. A `RequestMapper` function is a callable that takes the `Supervisor` object and the prompt to generate a dictionary as the JSON representation of the payload. If things are unclear, check out the implemented request mappers in `rest/request_mappers/`. The details of each request mapper are specific to the implementation.
All `RequestMapper` functions live in `rest/request_mappers/` in their own Python file.  The `rest/request_mappers/__init__.py` imports all request mappers directly from the files.

##### Step 4: The `AuthMapper` function
Most providers follow standardized authentication practices and use the implemented `auth_bearer` mapper. Some providers use custom ones, so for these, new mappers have to be implemented. An `AuthMapper` function takes a `Supervisor` object and returns a dictionary that is the JSON representation of the authentication header. As with `RequestMapper` functions, all authentication mappers live in `rest/auth_mappers/` in separate files and the `rest/auth_mappers/__init__.py` imports all authentication mappers directly from these files.

##### Step 5: The ResultMapper function
Since the POST response JSON will be represented as a dictionary, most result mappers will parse this dictionary to find the necessary flag value. This is different for every single model, but since it is just dictionary parsing, the task is usually straightforward.

##### Step 6: Auxiliary stuff
The REST supervisors are all quite customizable, so most of them require an arsenal of auxiliary attributes. In the same way, the authentications are all a bit different, so these tiny quirks will have to be accommodated by additional attributes (that is why the whole `Supervisor` object i spassed to the `RequestMapper` and `AuthMapper` functions).

##### Step 7: Add to the `AutoHuggingFaceSupervisor` mappings
At the top of the `supervisors/rest/auto_endpoint.py` there is a mapping dict called `MAPPING_DICT`. This dictionary maps a unique but semantic identifier of a supervision system to the submodule name in which it is implemented and the class name of its implementation. Make sure to add the newly implemented supervisor to this dictionary.