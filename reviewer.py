import json
import os
from pathlib import Path

from openai import AsyncOpenAI


class Reviewer:
    def __init__(self, model: str = "o4-mini", prompts_dir: str = "./prompts/paper_review"):

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.reviews = []

        self.reviewer_system = Path(prompts_dir+"/reviewer_system.txt").read_text()
        self.paper_review = Path(prompts_dir+"/paper_review.txt").read_text()
        self.neurips_reviewer_guidelines = Path(prompts_dir+"/neurips_reviewer_guidelines.txt").read_text()
        self.few_shot_review_examples = Path(prompts_dir+"/few_shot_review_examples.txt").read_text()
        self.paper_reflection = Path(prompts_dir+"/paper_reflection.txt").read_text()
        self.ensemble_system = Path(prompts_dir+"/ensemble_system.txt").read_text()



    async def review(self, paper_content: str, reflection: int = 3) -> list[dict]:
        

        messages = [{'role': 'system', 'content': self.reviewer_system}]

        paper_review = paper_review.replace("{neurips_reviewer_guidelines}", self.neurips_reviewer_guidelines)
        paper_review = paper_review.replace("{few_show_examples}", self.few_shot_review_examples)
        paper_review = paper_review.replace("{paper}", paper_content)

        prompt = paper_review

        messages.append({'role': 'user', 'content': prompt})

        # 1) 최초 리뷰 생성
        print("==> initial review generation start...")
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        print(completion.choices[0].message.content)

        try:
            review_json = json.loads(completion.choices[0].message.content)
        except json.JSONDecodeError:
            print("review JSON parsing failed")
            return

        self.reviews.append(review_json)
        messages.append({'role': 'assistant', 'content': completion.choices[0].message.content})
        print("==> initial review generation done.")

        # 2) 리뷰 리플렉션 생성
        for i in range(reflection):
            round_num = i + 1
            print(f"==> reflection {round_num}/{reflection} start...")
            messages.append({'role': 'user', 'content': f"Round {round_num}/{reflection}." + self.paper_reflection})

            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            try:
                reflection_json = json.loads(completion.choices[0].message.content)
            except json.JSONDecodeError:
                print(f"reflection {round_num} JSON parsing failed")
                return

            self.reviews.append(reflection_json)
            messages.append({'role': 'assistant', 'content': completion.choices[0].message.content})
            print(f"==> reflection {round_num}/{reflection} done.")

            if 'i am done' in completion.choices[0].message.content.lower():
                print("reflection ealy done")
                break

        return self.reviews

    async def review_ensembling(self) -> list[dict]:
        if not self.reviews:
            raise ValueError("No reviews available for ensembling.")

        print("==> review ensembling start...")
        
        self.ensemble_system = self.ensemble_system.replace("{reviewer_count}", str(len(self.reviews)))

        messages = [{'role': 'system', 'content': self.ensemble_system}]

        prompt = ""
        for idx, content in enumerate(self.reviews):
            review_text = json.dumps(content, indent=2) if isinstance(content, dict) else str(content)
            prompt += f"Review {idx + 1}/{len(self.reviews)}:\n{review_text}\n\n"

        prompt += "\n\n\n\n\n" + self.neurips_reviewer_guidelines
        messages.append({'role': 'user', 'content': prompt})

        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        try:
            final_review = json.loads(completion.choices[0].message.content)
        except json.JSONDecodeError:
            print("ensembling review JSON parsing failed")
            return

        self.reviews = [final_review]
        print("==> review ensembling done.")

        return self.reviews

    def is_review_strong_enough(self, score_threshold: float = 3.0, confidence_threshold: float = 3.0) -> bool:
        for review in self.reviews:
            if isinstance(review, dict):
                overall = review.get("overall_score")
                confidence = review.get("confidence")
                try:
                    if overall is not None and confidence is not None:
                        if float(overall) >= score_threshold and float(confidence) >= confidence_threshold:
                            return True
                except ValueError:
                    continue
        return False
 
