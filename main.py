from openai import OpenAI
from pathlib import Path
import json
import os

class Reviewr:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.reviews = []

    def review(self, paper_content: str, reflection: int = 3) -> list:
        reviewer_system = Path("./prompts/reviewer_system.txt").read_text()
        neurips_reviewer_guidelines = Path("./prompts/neurips_reviewer_guidelines.txt").read_text()
        few_shot_review_examples = Path("./prompts/few_shot_review_examples.txt").read_text()
        paper_reflection = Path("./prompts/paper_reflection.txt").read_text()

        messages = [{'role': 'system', 'content': reviewer_system}]

        prompt = f"""## Review Form
        Below is a description of the questions you will be asked on the review form for each paper and some guidelines on what to consider when answering these questions.
        When writing your review, please keep in mind that after decisions have been made, reviews and meta-reviews of accepted papers and opted-in rejected papers will be made public.


        {neurips_reviewer_guidelines}

        {few_shot_review_examples}

        Here is the paper you are asked to review:

        RESPONESE IS JSON FORMAT


        ```
        {paper_content}
        ```
        """

        messages.append({'role': 'user', 'content': prompt})

        # 1) 최초 리뷰 생성
        print("==> 최초 리뷰 생성 시작...")
        completion = self.client.chat.completions.create(
            model="o4-mini",
            messages=messages,
        )
        try:
            review_json = json.loads(completion.choices[0].message.content)
        except json.JSONDecodeError:
            print("리뷰 JSON 파싱 실패")
            return

        self.reviews.append(review_json)
        messages.append({'role': 'assistant', 'content': completion.choices[0].message.content})
        print("==> 최초 리뷰 생성 완료.")

        # 2) 리뷰 리플렉션 생성
        for i in range(reflection):
            round_num = i + 1
            print(f"==> 리플렉션 {round_num}/{reflection} 시작...")
            messages.append({'role': 'user', 'content': f"Round {round_num}/{reflection}." + paper_reflection})

            completion = self.client.chat.completions.create(
                model="o4-mini",
                messages=messages,
            )
            try:
                reflection_json = json.loads(completion.choices[0].message.content)
            except json.JSONDecodeError:
                print(f"리플렉션 {round_num} JSON 파싱 실패")
                return

            self.reviews.append(reflection_json)
            messages.append({'role': 'assistant', 'content': completion.choices[0].message.content})
            print(f"==> 리플렉션 {round_num}/{reflection} 완료.")

            if 'i am done' in completion.choices[0].message.content.lower():
                print("리플렉션 루프 중단 요청 감지, 종료합니다.")
                break

        return self.reviews

    def review_ensembling(self) -> list:
        if not self.reviews:
            raise ValueError("No reviews available for ensembling.")

        print("==> 리뷰 앙상블 시작...")
        neurips_reviewer_guidelines = Path("./prompts/neurips_reviewer_guidelines.txt").read_text()
        ensemble_system = Path("./prompts/ensemble_system.txt").read_text()
        ensemble_system = ensemble_system.replace("{reviewer_count}", str(len(self.reviews)))

        messages = [{'role': 'system', 'content': ensemble_system}]

        # 앙상블 전처리
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
            print("앙상블 리뷰 JSON 파싱 실패")
            return

        self.reviews = [final_review]
        print("==> 리뷰 앙상블 완료.")
        return self.reviews

    def is_review_strong_enough(self, score_threshold: float = 4.0, confidence_threshold: float = 3.0) -> bool:
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

if __name__ == "__main__":
    # Example usage
    reviewr = Reviewr()
    paper_content = Path("./paper.txt").read_text()

    reviews = reviewr.review(paper_content, reflection=3)
    print(reviews)

    ensemble_reviews = reviewr.review_ensembling()
    print(ensemble_reviews)

    is_strong_enough = reviewr.is_review_strong_enough()
    print(is_strong_enough)
