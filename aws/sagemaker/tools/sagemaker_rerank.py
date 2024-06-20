import boto3
import json

from typing import Any, Optional, Union, List
from core.tools.entities.tool_entities import ToolInvokeMessage
from core.tools.tool.builtin_tool import BuiltinTool


class SageMakerReRankTool(BuiltinTool):
    sagemaker_client: Any = None
    sagemaker_endpoint:str = None
    topk:int = None

    def _sagemaker_rerank(self, query_input: str, docs: List[str], rerank_endpoint:str):
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

    def _invoke(self, 
                user_id: str, 
               tool_parameters: dict[str, Any], 
        ) -> Union[ToolInvokeMessage, list[ToolInvokeMessage]]:
        """
            invoke tools
        """
        line = 0
        try:
            if not self.sagemaker_client:
                access_key = self.runtime.credentials.get('aws_access_key_id', None)
                secret_key = self.runtime.credentials.get('aws_secret_access_key', None)
                aws_region = self.runtime.credentials.get('aws_region', None)
                self.sagemaker_client = boto3.client("sagemaker-runtime", region_name=aws_region, aws_access_key_id=access_key, aws_secret_access_key=secret_key)

            line = 1
            if not self.sagemaker_endpoint:
                self.sagemaker_endpoint = self.runtime.credentials.get('sagemaker_endpoint', None)

            line = 2
            if not self.topk:
                self.topk = self.runtime.credentials.get('topk', 5)

            line = 3
            query = tool_parameters.get('query', '')
            if not query:
                return self.create_text_message('Please input query')
            
            line = 4
            candidate_texts = tool_parameters.get('candidate_texts', '')
            if not candidate_texts:
                return self.create_text_message('Please input candidate_texts')
            
            line = 5
            candidate_docs = candidate_texts.split('$$$')

            line = 6
            scores = self._sagemaker_rerank(query_input=query, docs=candidate_docs, rerank_endpoint=self.sagemaker_endpoint)

            line = 7
            sorted_scores = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)

            line = 8
            results = [ item[0] for item in sorted_scores[:self.topk]]

            line = 9
            return [ self.create_text_message(content=result) for result in results ]
        except Exception as e:
            return self.create_text_message(f'Exception {str(e)}, line : {line}')
    