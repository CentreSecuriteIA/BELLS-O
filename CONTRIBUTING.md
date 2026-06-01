# Contributing to BELLS-O

Thanks for helping extend BELLS-O! This guide covers how to add new **supervision systems**
(guardrails). For an overview of the framework, see the [README](https://github.com/CentreSecuriteIA/BELLS-O#readme).

## Dev setup

```bash
uv sync --extra dev          # or: pip install -e ".[dev]"
```

Formatting and linting are handled by [ruff](https://docs.astral.sh/ruff/); the configuration
lives in `pyproject.toml` (line length 120, `py312` target). Please run ruff before opening a PR:

```bash
uvx ruff format . && uvx ruff check .
```

> TODO: pre-commit hooks for ruff are planned — once added, enable them with `pre-commit install`.

## Anatomy of a supervisor

A supervision system is defined by four things:

1. A **base class** to extend — `HuggingFaceSupervisor`, `RestSupervisor`, or `CustomSupervisor`.
2. A **pre-processor** that maps a prompt into the system's expected input format.
3. A **`ResultMapper`** that maps the system's raw output into a `Result` object.
4. Any **auxiliary** parameters or functions specific to that system.

### Which base class?

- A model you run yourself from HuggingFace → `HuggingFaceSupervisor`.
- A hosted HTTP endpoint → `RestSupervisor`.
- Anything that needs a bespoke client library (e.g. ProtectAI LLM Guard) → `CustomSupervisor`
  (see `src/bells_o/supervisors/custom/`).

### What is a `ResultMapper`?

A `ResultMapper` is a callable `(<raw_output>, usage: Usage) -> Result`. For HuggingFace
supervisors the raw output is the decoded string; for REST supervisors it is the parsed JSON
response (a dict). The mapper inspects that output and sets the boolean for each task type in the
returned `Result`.

The `usage` argument is always passed but can usually be ignored, since a mapper is typically
written for one specific model that supports a single task type.

Mappers are also responsible for **multi-category flagging** and **float-to-bool conversion**:

- When a model reports a probability, the default flagging threshold is **0.85**, unless the
  model's documentation specifies otherwise (documented thresholds take priority).
- If a model scores multiple categories, **any one** category crossing the threshold makes the
  prompt harmful.

All `ResultMapper` functions live in `src/bells_o/result_mappers/`, one file per system (no
submodules). `result_mappers/__init__.py` imports every mapper directly from its file.

## Implementing a HuggingFace supervisor

**1. Module structure.** Modules are `bells_o.supervisors.huggingface.<lab_name>.<model_name>`.
Each lab's `__init__.py` imports its supervisor classes, and `huggingface/__init__.py` imports the
classes from every lab submodule.

**2. `__init__` and attributes.** The constructor must accept at least `pre_processing`,
`model_kwargs`, `tokenizer_kwargs`, and `generation_kwargs`, and must set:

- `self.name` — the exact HuggingFace model id (used to load the model/tokenizer).
- `self.usage` — a `Usage` object declaring the supported task type(s).
- `self.res_map_fn` — the result mapper for this supervisor.
- forward the constructor arguments onto `self.pre_processing`, `self.model_kwargs`,
  `self.tokenizer_kwargs`, `self.generation_kwargs`.

**3. Input formatting.** For most models the `RoleWrapper` pre-processor is enough — see its
docstring in `preprocessors/role_wrapper.py`. Append it to the `pre_processing` list before setting
the attribute. For a more involved setup, see `huggingface/saillab/xguard_supervisor.py`.

**4. The `ResultMapper`.** The decoded output is a string, so most mappers regex-parse it for a
flag and extract its value. The exact format differs for every model — be ready to be surprised.

**5. Auxiliary needs.** Some models need extra constructor arguments or non-standard input formats.
See `huggingface/openai/gpt_oss.py` for an example.

**6. Register it.** Add an entry to `MODEL_MAPPING` at the top of
`src/bells_o/supervisors/huggingface/auto_model.py`. It maps the HuggingFace model id to a tuple of
`(submodule_name, class_name, init_kwargs)` so `AutoHuggingFaceSupervisor.load(...)` can find it.

## Implementing a REST supervisor

**1. Module structure.** Modules are `bells_o.supervisors.rest.<provider_name>.<endpoint_name>`.
Each provider's `__init__.py` imports its classes, and `rest/__init__.py` imports them from every
provider submodule.

**2. `__init__` and attributes.** The constructor must accept at least `pre_processing`, `api_key`,
and `api_variable`, and must set:

- `self.name` — the supervisor name.
- `self.provider_name` — the provider name.
- `self.base_url` — the REST base URL.
- `self.usage` — a `Usage` object.
- `self.res_map_fn` — the result mapper.
- `self.req_map_fn` — a `RequestMapper` that builds the POST body.
- `self.auth_map_fn` — an `AuthMapper` that builds the auth header.
- forward `self.pre_processing`, `self.api_key`, `self.api_variable`.
- optionally `self.custom_header` — extra headers merged with the auth header (some endpoints need
  more than authentication).

**3. The `RequestMapper`.** REST payloads are non-standardized, so each system has its own. A
`RequestMapper` is `(supervisor, prompt) -> dict` (the JSON body). See existing examples in
`rest/request_mappers/`; each file holds one mapper, and `request_mappers/__init__.py` imports them
all.

**4. The `AuthMapper`.** Most providers use bearer tokens via the shared `auth_bearer` mapper. For
others, add a new mapper — `(supervisor) -> dict` (the auth header). Auth mappers live one-per-file
in `rest/auth_mappers/`, imported by `auth_mappers/__init__.py`.

**5. The `ResultMapper`.** The response JSON is a dict, so the mapper usually just walks the dict to
find the relevant flag — generally straightforward.

**6. Auxiliary needs.** REST supervisors are highly customizable; the whole `Supervisor` object is
passed to the request and auth mappers, so per-endpoint quirks can be accommodated with extra
attributes.

**7. Register it.** Add an entry to `MODEL_MAPPING` at the top of
`src/bells_o/supervisors/rest/auto_endpoint.py`, mapping a unique, semantic endpoint id to
`(submodule_name, class_name)` so `AutoRestSupervisor.load(...)` can find it.

## Implementing a custom supervisor

For systems that need their own client library, extend `CustomSupervisor`
(`src/bells_o/supervisors/custom/custom_supervisor.py`) and register it in
`custom/auto_model.py`'s `MODEL_MAPPING`, loadable via `AutoCustomSupervisor.load(...)`. See
`custom/protectai/protectai_llm_guard.py` for a worked example.
