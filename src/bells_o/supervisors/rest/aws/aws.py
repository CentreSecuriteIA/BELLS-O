"""Implement the AWS Bedrock base supervisor via boto3."""

from functools import partial
from os import getenv
from time import sleep, time
from typing import Any, cast


try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    boto3 = None  # type: ignore
    ClientError = Exception  # type: ignore
    NoCredentialsError = Exception  # type: ignore

from bells_o.common import AuthMapper, OutputDict, RequestMapper, ResultMapper, Usage
from bells_o.preprocessors import PreProcessing

from ..auth_mappers import aws as aws_auth_map
from ..request_mappers import aws as aws_request_map
from ..rest_supervisor import RestSupervisor


class AwsSupervisor(RestSupervisor):
    """Base class for AWS Bedrock supervisors using boto3."""

    def __init__(
        self,
        name: str,
        usage: Usage,
        result_mapper: ResultMapper,
        base_url: str = "",
        region: str = "us-east-1",
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "AWS_ACCESS_KEY_ID",
        source: str = "INPUT",
    ):
        """Initialize the AwsSupervisor.

        Args:
            name (str): Name/identifier for the supervisor (e.g., guardrail ID).
            usage (Usage): The usage type of the supervisor.
            result_mapper (ResultMapper): ResultMapper to use for this Supervisor.
            base_url (str): Base URL of the API endpoint (not used with boto3, kept for compatibility).
            region (str): AWS region. Defaults to "us-east-1".
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key (str | None, optional): AWS Access Key ID (optional, can use IAM role or credentials file). Defaults to None.
            api_variable (str, optional): Environment variable name for AWS Access Key ID. Defaults to "AWS_ACCESS_KEY_ID".
            source (str, optional): Content source, either "INPUT" or "OUTPUT". Defaults to "INPUT".

        """
        if boto3 is None:
            raise ImportError("boto3 is required for AWS supervisors. Install it with: pip install boto3")

        self.name: str = name
        self.provider_name: str | None = "AWS"
        self.custom_header = {}
        self.base_url: str = base_url
        self.usage: Usage = usage
        self.res_map_fn: ResultMapper = cast(ResultMapper, partial(result_mapper, usage=self.usage))
        self.req_map_fn: RequestMapper = aws_request_map
        self.auth_map_fn: AuthMapper = aws_auth_map
        self.pre_processing = pre_processing
        self.region: str = region
        self.source: str = source
        self.api_key = api_key
        self.api_variable = api_variable
        self.needs_api = False  # AWS uses boto3 credentials, not API key in headers

        # Initialize boto3 client
        self._bedrock_client = None

        super().__post_init__()

    @property
    def bedrock_client(self):
        """Get or create the boto3 bedrock-runtime client."""
        if self._bedrock_client is None:
            # boto3 automatically uses the default credential chain:
            # 1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
            # 2. AWS credentials file (~/.aws/credentials)
            # 3. IAM role (if running on EC2/ECS/Lambda)
            try:
                self._bedrock_client = boto3.client("bedrock-runtime", region_name=self.region)
            except NoCredentialsError:
                # Provide helpful error message
                access_key = getenv("AWS_ACCESS_KEY_ID")
                secret_key = getenv("AWS_SECRET_ACCESS_KEY")

                error_msg = (
                    "AWS credentials not found. Please configure credentials using one of:\n"
                    "1. Environment variables: Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY\n"
                    "2. AWS credentials file: ~/.aws/credentials\n"
                    "3. IAM role (if running on AWS infrastructure)\n\n"
                )

                if not access_key and not secret_key:
                    error_msg += (
                        "Tip: Create a .env file with:\n"
                        "AWS_ACCESS_KEY_ID=your_access_key\n"
                        "AWS_SECRET_ACCESS_KEY=your_secret_key\n"
                        "And load it with: from dotenv import load_dotenv; load_dotenv()"
                    )

                raise NoCredentialsError(error_msg)
        return self._bedrock_client

    def _judge_sample(
        self,
        prompt: str,
    ) -> OutputDict:
        """Run an individual API request for inference using boto3.

        Args:
            prompt (str): The prompt string to check.

        Returns:
            OutputDict: The output of the Supervisor and corresponding metadata.

        """
        tried_once = False
        no_valid_response = True

        while no_valid_response:
            if tried_once:
                print("INFO: Retrying generation in 5s. Hit rate limit.")
                sleep(5)
            else:
                print("INFO: Generating judgement.")

            start_time = time()

            try:
                # Prepare request payload
                request_payload = self.req_map_fn(self, prompt)

                # Call AWS Bedrock Guardrail API via boto3
                # This will be overridden in subclasses for specific endpoints
                response = self._call_bedrock_api(request_payload)
                generation_time = time() - start_time

                tried_once = True
                no_valid_response = False

            except (ClientError, NoCredentialsError) as e:
                generation_time = time() - start_time
                # Handle credential errors
                if isinstance(e, NoCredentialsError):
                    raise
                # Handle rate limiting
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code == "ThrottlingException" or error_code == "TooManyRequestsException":
                    no_valid_response = True
                    tried_once = True
                else:
                    # Re-raise other errors
                    raise

        return OutputDict(output_raw=response, metadata={"latency": generation_time})

    def _call_bedrock_api(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        """Call the AWS Bedrock API. To be overridden by subclasses.

        Args:
            request_payload: The request payload.

        Returns:
            dict[str, Any]: The API response.

        """
        raise NotImplementedError("Subclasses must implement _call_bedrock_api")
