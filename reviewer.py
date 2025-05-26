from openai import OpenAI
from pathlib import Path
import json
import os

class Reviewer:
    def __init__(self, model: str = "o4-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.reviews = []

    def review(self, paper_content: str, reflection: int = 3) -> list[dict]:
        reviewer_system = Path("./prompts/reviewer_system.txt").read_text()
        paper_review = Path("./prompts/paper_review.txt").read_text()
        neurips_reviewer_guidelines = Path("./prompts/neurips_reviewer_guidelines.txt").read_text()
        few_shot_review_examples = Path("./prompts/few_shot_review_examples.txt").read_text()
        paper_reflection = Path("./prompts/paper_reflection.txt").read_text()

        messages = [{'role': 'system', 'content': reviewer_system}]

        paper_review = paper_review.replace("{neurips_reviewer_guidelines}", neurips_reviewer_guidelines)
        paper_review = paper_review.replace("{few_show_examples}", few_shot_review_examples)
        paper_review = paper_review.replace("{paper}", paper_content)

        prompt = paper_review

        messages.append({'role': 'user', 'content': prompt})

        # 1) 최초 리뷰 생성
        print("==> initial review generation start...")
        completion = self.client.chat.completions.create(
            model="o4-mini",
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
            messages.append({'role': 'user', 'content': f"Round {round_num}/{reflection}." + paper_reflection})

            completion = self.client.chat.completions.create(
                model="o4-mini",
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

    def review_ensembling(self) -> list[dict]:
        if not self.reviews:
            raise ValueError("No reviews available for ensembling.")

        print("==> review ensembling start...")
        neurips_reviewer_guidelines = Path("./prompts/neurips_reviewer_guidelines.txt").read_text()
        ensemble_system = Path("./prompts/ensemble_system.txt").read_text()
        ensemble_system = ensemble_system.replace("{reviewer_count}", str(len(self.reviews)))

        messages = [{'role': 'system', 'content': ensemble_system}]

        prompt = ""
        for idx, content in enumerate(self.reviews):
            review_text = json.dumps(content, indent=2) if isinstance(content, dict) else str(content)
            prompt += f"Review {idx + 1}/{len(self.reviews)}:\n{review_text}\n\n"

        prompt += "\n\n\n\n\n" + neurips_reviewer_guidelines
        messages.append({'role': 'user', 'content': prompt})

        completion = self.client.chat.completions.create(
            model="o4-mini",
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
