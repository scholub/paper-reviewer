from pathlib import Path

from .reviewer import Reviewer

if __name__ == "__main__":
    paper_content = Path("./paper.txt").read_text()

    r1 = Reviewer(model="o4-mini")
    r1.review(paper_content, reflection=3)
    ensemble_reviews = r1.review_ensembling()
    print(ensemble_reviews)

    r2 = Reviewer(model="o4-mini")
    r2.review(paper_content, reflection=3)
    ensemble_reviews2 = r2.review_ensembling()
    print(ensemble_reviews)
    
    r3 = Reviewer(model="o4-mini")
    r3.review(paper_content, reflection=3)
    ensemble_reviews2 = r3.review_ensembling()
    print(ensemble_reviews)

    chair = Reviewer(model="o4-mini")
    chair.reviews = r1.reviews + r2.reviews + r3.reviews
    chair_ensemble_reviews = chair.review_ensembling()
    print(chair_ensemble_reviews)
    print("Is review strong enough?", chair.is_review_strong_enough())

    

    


