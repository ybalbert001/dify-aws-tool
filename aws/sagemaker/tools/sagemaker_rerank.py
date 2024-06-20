from typing import Any, Optional, Union
from core.tools.entities.tool_entities import ToolInvokeMessage
from core.tools.tool.builtin_tool import BuiltinTool


class SageMakerReRankTool(BuiltinTool):
    sagemaker_client = None
    sagemaker_endpoint = None
    topk = None

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
        if not self.sagemaker_client:
            access_key = self.runtime.credentials.get('aws_access_key_id', None)
            secret_key = self.runtime.credentials.get('aws_secret_access_key', None)
            aws_region = self.runtime.credentials.get('aws_region', None)
            self.sagemaker_client = boto3.client("sagemaker-runtime", region_name=aws_region, aws_access_key_id=access_key, aws_secret_access_key=secret_key)

        if not self.sagemaker_endpoint:
            self.sagemaker_endpoint = self.runtime.credentials.get('sagemaker_endpoint', None)

        if not self.topk:
            self.topk = self.runtime.credentials.get('topk', 5)

        query = tool_parameters.get('query', '')
        if not query:
            return self.create_text_message('Please input query')
        
        candidate_texts = tool_parameters.get('candidate_texts', '')
        if not candidate_texts:
            return self.create_text_message('Please input candidate_texts')
        
        candidate_docs = candidate_texts.split('$$$')

        scores = self._sagemaker_rerank(query_input=query, docs=candidate_docs, rerank_endpoint=self.sagemaker_endpoint)

        sorted_scores = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)

        results = [ item[0] for item in sorted_scores[:self.topk]]

        return [ self.create_text_message(self.summary(user_id=user_id, content=result)) for result in results ]
    