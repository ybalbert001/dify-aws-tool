from typing import Optional, List

from core.external_data_tool.base import ExternalDataTool
import boto3

class TextListRerank(ExternalDataTool):
    """
    The name of custom type must be unique, keep the same with directory and file name.
    """
    name: str = "sagemaker_rerank"
    sagemaker_client = None

    @classmethod
    def validate_config(cls, tenant_id: str, config: dict) -> None:
        """
        schema.json validation. It will be called when user save the config.

        Example:
            .. code-block:: python
                config = {
                    "temperature_unit": "centigrade"
                }

        :param tenant_id: the id of workspace
        :param config: the variables of form config
        :return:
        """

        if not config.get('api_endpoint'):
            raise ValueError('api_endpoint is required')
        if not config.get('aws_ak'):
            raise ValueError('aws_ak is required')
        if not config.get('aws_sk'):
            raise ValueError('aws_sk is required')
        if not config.get('aws_region'):
            raise ValueError('aws_region is required')

    def sagemaker_rerank(self, query_input: str, docs: List[str], rerank_endpoint:str):
        inputs = [query_input]*len(docs)
        response_model = self.sagemaker_client.invoke_endpoint(
            EndpointName=rerank_endpoint,
            Body=json.dumps(
                {
                    "inputs": inputs,
                    "docs": [item['doc'] for item in docs]
                }
            ),
            ContentType="application/json",
        )
        json_str = response_model['Body'].read().decode('utf8')
        json_obj = json.loads(json_str)
        scores = json_obj['scores']
        return scores if isinstance(scores, list) else [scores]

    def rerank(self, query_input: str, docs: List[str], topk:int) -> list:
        """
        Query the external data tool.

        :param inputs: user inputs
        :param query: the query of chat app
        :return: the tool query result
        """
        if not sagemaker_client:
            aws_ak = self.config.get('aws_ak')
            aws_sk = self.config.get('aws_ak')
            aws_region = self.config.get('aws_ak')
            self.sagemaker_client = boto3.client("sagemaker-runtime", )

        scores = self.sagemaker_rerank(query_input, docs, rerank_endpoint)
        sorted_scores = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return sorted_scores[:topk]