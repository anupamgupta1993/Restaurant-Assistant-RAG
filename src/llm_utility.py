from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from time import time
from ingest import load_index
import json
import os

class RAGQueryEngine:
    def __init__(self):
        openai_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=openai_key)
        self.model = "gpt-5-nano"
        self.prompt_template = """
                    You're a restaurant connoisseur. Answer the QUESTION based on the CONTEXT from our restaurant and menu items database.
                    Use only the facts from the CONTEXT when answering the QUESTION.

                    QUESTION: {question}

                    CONTEXT:
                    {context}
                    """.strip()
        self.record_template = """
                    restaurant_name: {name_x}
                    score: {score}
                    ratings_count: {ratings}
                    restaurant_category: {category_x}
                    price_range: {price_range}
                    full_address: {full_address}
                    zip_code: {zip_code}
                    lat: {lat}
                    lng: {lng}
                    restaurant_id: {restaurant_id}
                    menu_category: {category_y}
                    menu_item_name: {name_y}
                    description: {description}
                    item_price: {price}
                    city: {city}
                    state: {state}
                    """.strip()
        self.evaluation_prompt_template = """
            You are an expert evaluator for a RAG system.
            Your task is to analyze the relevance of the generated answer to the given question.
            Based on the relevance of the generated answer, you will classify it
            as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

            Here is the data for evaluation:

            Question: {question}
            Generated Answer: {answer}

            Please analyze the content and context of the generated answer in relation to the question
            and provide your evaluation in parsable JSON without using code blocks:

            {{
            "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
            "Explanation": "[Provide a brief explanation for your evaluation]"
            }}
            """.strip()
        self.retrieval_index = load_index()

    def _build_context(self, search_results: List[Dict]) -> str:
        """Formats documents into the record template."""
        context_blocks = [
            self.record_template.format(**doc) for doc in search_results
        ]
        return "\n\n".join(context_blocks)

    def build_prompt(self, query: str, search_results: List[Dict]) -> str:
        """Constructs the final LLM prompt."""
        context = self._build_context(search_results)
        return self.prompt_template.format(question=query, context=context).strip()

    def query_llm(self, query: str, search_results: List[Dict]) -> Dict:
        """Formats a prompt, queries the LLM, and returns answer + token stats."""
        prompt = self.build_prompt(query, search_results)
        
        t0 = time()
        ans, token_stats = self.llm(prompt)

        relevance, rel_token_stats = self.evaluate_relevance(query, ans)
        t1 = time()
        took = t1 - t0

        openai_cost_rag = self.calculate_openai_cost(token_stats)
        openai_cost_eval = self.calculate_openai_cost(rel_token_stats)

        openai_cost = openai_cost_rag + openai_cost_eval

        answer_data = {
            "answer": ans,
            "model_used": self.model,
            "response_time": took,
            "relevance": relevance.get("Relevance", "UNKNOWN"),
            "relevance_explanation": relevance.get(
                "Explanation", "Failed to parse evaluation"
            ),
            "prompt_tokens": token_stats["prompt_tokens"],
            "completion_tokens": token_stats["completion_tokens"],
            "total_tokens": token_stats["total_tokens"],
            "eval_prompt_tokens": rel_token_stats["prompt_tokens"],
            "eval_completion_tokens": rel_token_stats["completion_tokens"],
            "eval_total_tokens": rel_token_stats["total_tokens"],
            "openai_cost": openai_cost,
        }

        return answer_data
        

    def evaluate_relevance(self, question, answer):
        prompt = self.evaluation_prompt_template.format(question=question, answer=answer)
        evaluation, tokens = self.llm(prompt)

        try:
            json_eval = json.loads(evaluation)
            return json_eval, tokens
        except json.JSONDecodeError:
            result = {"Relevance": "UNKNOWN", "Explanation": "Failed to parse evaluation"}
            return result, tokens

    def calculate_openai_cost(self, tokens):
        # the below cost is for gpt 5 nano as that is the only model i am using in this project
        cost = tokens["prompt_tokens"] * 0.00000005 + tokens["completion_tokens"] * 0.0000004
        return cost

    def llm(self, prompt: str) -> Tuple[str, Dict]:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.choices[0].message.content
        usage = response.usage

        token_stats = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

        return answer, token_stats
        
