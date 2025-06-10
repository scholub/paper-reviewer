from pathlib import Path

from reviewer import Reviewer
import asyncio

if __name__ == "__main__":
    async def main():
        paper_content = Path("./paper.txt").read_text()

        r1 = Reviewer(model="o4-mini")
        await r1.review(paper_content, reflection=3)
        ensemble_reviews = await r1.review_ensembling()
        print(ensemble_reviews)

    asyncio.run(main())
