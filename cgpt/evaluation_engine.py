"""Interview evaluation utilities for step-by-step scoring.

Implements the Platform Evaluation Rubric and Business Logic (October 2025)
Dimensions: Technical Reasoning, Accuracy, Confidence, Problem-Solving, Flow/Stuckness.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv(), override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLIENT: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

RUBRIC_WEIGHTS = {
    "technical_reasoning": 0.30,
    "accuracy": 0.25,
    "confidence": 0.15,
    "problem_solving": 0.15,
    "flow": 0.15,
}

SKIP_NOTE = "Icebreaker question — skipped for scoring."

REFERENCE_EVALUATION_EXAMPLE = """{
    "interview_id": "mock_001",
    "candidate_name": "Anuja Tipare",
    "job_role": "Data Scientist",
    "interview_date": "2025-05-21",
    "questions": [
        {
            "id": 1,
            "question": "Hi Anuja, how are you?",
            "answer": "I'm doing good. How are you?",
            "feedback": "This is just an introduction question. No feedback needed.",
            "score": null,
            "improvement_areas": [],
            "type": "introduction",
            "followup_to": null
        },
        {
            "id": 2,
            "question": "What we'll do is we'll have a few questions from the resume from any of the projects that you worked on. We'll try to focus on the basics of machine learning data science. So to start off with, describe any one particular project that you've worked on recently, primarily involving some form of data science or ML if possible, any particular project and you can go from the problem statement to what you did and how, you know, sort of what the results were or if you had, if you created some business impact.",
            "answer": "Definitely Ankit. So last quarter I worked on a recommendation system for a bookstore. So the problem statement was that the bookstore currently was recommending books by top selling items and it wasn't personalized for the user. So what I and my team did was we created a personalized recommendation system which was based on the recency, frequency and monetary value of the data that we got from. So the data had around 30 book categories for 33k customers and we had the recency, frequency, monetary values and as it is very popular in the marketing segment, we used the RFM to create clusters. So we did two kinds of clustering. The first was the customer value segmentation based on the RFM values which basically gave me four clusters of my customers. Some were the potential loyalists, some were the champions who had very good high RFM values and there were lapsed customers who had low RFM values and that was the first clustering that we did. Within the group style, just explain me, did clustering based on the book categories. Now, as I said, we had somewhere around 30 book categories, but we clubbed them together like history and something subjective was clubbed together. So based on this, we did another clustering, both of which was done using k-means. Once the clusters were ready, we implemented collaborative filtering for both user-based and item-based filtering. To get to see how the personalized recommendation system worked, we implemented an A-B testing framework. So we had a test group and a control group and then we tracked some KPIs like the click through rate, conversion rate and the average session time. So the metrics over the silhouette score, which was 0.44 were kind of okay. So it performed moderately. Yeah, that was the project that I did.",
            "feedback": "You communicated your project experience with confidence and covered the full lifecycle from problem statement to evaluation. This is a strong example of how to describe a project.",
            "score": 8.5,
            "improvement_areas": ["Discuss trade-offs or challenges faced","Explain terms like RFM, collaborative filtering, and silhouette score briefly for clarity"],
            "type": "technical",
            "followup_to": null
        },
        {
            "id": 3,
            "question": "Great. So it sounds like that you had like a two-step approach. First, you were clustering the users and then you were clustering the books and the items, like the purchase. Okay. So you used k-means that I can see. Can you just tell me, so when we do k-means clustering, we prioritize normalizing the features or scaling the features. Do you know why is that important?",
            "answer": "So how k-means works is that randomly centroids are allocated between the data points and then it just calculates the mean of the group, the groups that are present in the centroids and then it gradually updates. Now if there are certain outliers, it is almost very much possible that your centroids or your means will get, they will be disrupted. It will be more inclined towards the outline and we don't want that. So for that purpose, it is necessary that we always normalize or bring it to a uniform scale before we perform k-means.",
            "feedback": "You rightly mentioned that unequal feature scales and outliers can skew centroids, which is a key insight. However, the explanation could benefit from clearer phrasing and a more direct focus on why scaling is important, specifically, because k-means relies on Euclidean distance, which is sensitive to feature magnitudes. You can also give an example here, like having features such as height in meters and weight in grams. Without scaling, the weight in grams would completely overshadow the height in meters in any distance calculation.",
            "score": 7,
            "improvement_areas": ["Emphasize that k-means being sensitive to feature scale",""],
            "type": "technical",
            "followup_to": 2
        },
        {
            "id": 4,
            "question": "Got it. Can you tell me some basic differences between k-means and like some other hierarchical clustering methods or like dbscan also for that matter? Can you just tell me some basic differences between these?",
            "answer": "So for k-means, we partition the data into clusters. We define the number of clusters that we want and with dbscan, it is a density based clustering algorithm and in k-means, we define k and dbscan, we define the distance, the minimum distance within which the points will be clustered and also the minimum number of points that we want for it to become a cluster. So the difference between this is that, as I said, that dbscan is more utilized to find the outliers because there is no certain, it will just group the points that are close together so it is good at anticipating the outliers. K-means does not work that way. And we also have hierarchical clustering which is, it basically clusters the entire data group. Either you start with a top down approach or the bottom up approach and in both of the approaches, you either start with the entire data set, one cluster and then divide it until you get to the last data point or you start in the reverse way. You start with one data point and then cluster everything into one single cluster and you can, depending on the length, the depth, you can cut it off anywhere and you will have the desired number of clusters. So these are some different kinds of clustering methods. We also have GMM, which is Gaussian mixture model, which doesn't look at hard probabilities, but it's like a soft divisioning algorithm.",
            "feedback": "You did a good job mentioning multiple clustering algorithms and highlighting some key differences like k-means being centroid-based, DBSCAN being density-based, and hierarchical using agglomerative/divisive approaches. The inclusion of GMM was a nice touch that shows broader awareness. You can further focus on criteria like assumptions, input parameters, ability to handle noise, and cluster shapes",
            "score": 8,
            "improvement_areas": ["Mention k-means struggles with non-convex clusters; DBSCAN handles them better","K-Means is generally more scalable to large datasets than standard hierarchical clustering or DBSCAN"],
            "type": "technical",
            "followup_to": 2
        },
        {
            "id": 5,
            "question": "Got it. Yeah. I think that pretty much answers my question. Last question on this front would be, so you said you kind of formulated an A-B testing design. Do you know how basic A-B testing works? How do you split into test and control groups?",
            "answer": "Yeah. So A-B testing is again, we do A-B testing to see if I'm implementing a new thing, how much effective is it. So in A-B testing, there are basically, there's a test group and control group. So first, the control group is what we, we don't introduce that control group with the new feature that we are implementing, but we introduce a new feature to the test group. And then we look at some of the metrics that we want to test. And there's a null hypothesis. There's a null hypothesis that, where the null hypothesis states that the new implementation has no effect on the metric whatsoever. Now your metric could be sales, it could be click through rate, it can be anything. So the null hypothesis states that there was no change in the result. And then you have the alternate hypothesis that it had some kind of impact. So then you run the test and then you compare. And our aim obviously is to reject the null hypothesis because we want to see if it had any impact or not. But given in real life scenarios, only having an impact is not the only factor that we consider. We want to see that implementing the change that you want to do, it converts into something profitable.",
            "feedback": "Good job on mentioning the ultimate goal being profitability or positive business impact is also a great practical touch. However, the explanation would be stronger with more explanation around randomization, statistical significance, and what it means to reject the null hypothesis.",
            "score": 8.5,
            "improvement_areas": ["Talk about the Mechanism of Splitting (random assignment)","Define terms like p-value and significance level (e.g., alpha = 0.05)"],
            "type": "project | technical",
            "followup_to": 2
        },
        {
            "id": 6,
            "question": "Correct. And lastly, on this, do you know like one or two tests that we can do, like statistical tests that we can do to see if we want to reject or fail to reject a null hypothesis?",
            "answer": "Yeah, we have different kinds of tests. We have the t-test, the z-test. So the t-test is a t-distribution test and it is used for longer tails. And when the data sample is less than 30, we go with t-test. If it's greater than 30, then we go with z-test. These are just statistical tests which can be around the value of alpha. We can clarify if it passes or not.",
            "feedback": "The t-test is robust even for larger samples if the population standard deviation is unknown. Try to give more context of what difference do these tests measure (mean of samples). Also talk briefly about other tests such as chi-squared test, and also try to touch up on parametric and non-parametric tests",
            "score": 7,
            "improvement_areas": ["Explain the role of the p-value in decision-making","Briefly mention one-tailed vs. two-tailed tests if time permits"],
            "type": "technical",
            "followup_to": 5
        },
        {
            "id": 7,
            "question": "Alright, so that was more on the project. Can we just move on to like few basic ML questions which would be useful for our use cases in our company? What do you feel the most comfortable with regression, classification? I mean, totally up to you.",
            "answer": "Anything is fine. Classification.",
            "feedback": "This is a clarification question. No feedback needed",
            "score": null,
            "improvement_areas": [],
            "type": "clarification",
            "followup_to": null
        },
        {
            "id": 8,
            "question": "Okay, let's go with classification then. Alright, can you tell me what is the difference between precision and recall?",
            "answer": "Okay, so precision and recall goes this way that precision is out of everything that you have predicted, how much is actually true. And recall is out of everything that is true, how much have you actually predicted. So giving you an example, let's say you have to predict if a person churns or not and you predicted that 800 people churned but let's say only 700 actually churned. So you were wrong about the other 100. So that wouldn't be a precision. And let's say there were a total of 850 that churned but you were able to predict only 600 or 650 then that would be recall. So it's the trade-off between the true positives and false positives.",
            "feedback": "Great job on explaining the concepts. The churn example you provided is also excellent for illustrating the concepts concretely. However, the explanation could be more precise with definitions based on true positives, false positives, and false negatives.",
            "score": 8.5,
            "improvement_areas": ["Try to use formal definitions of TP, FP, etc.","Mention their use in different contexts (example: spam detection, healthcare, etc.)"],
            "type": "technical",
            "followup_to": null
        },
        {
            "id": 9,
            "question": "Okay, so let's say you talked about a use case, a churn use case. So what will you think of prioritizing more here, recall or precision? Like if given that churned customers lead to some revenue loss, what would you try to sort of, you know, improve like, or like what would you try to focus on?",
            "answer": "Okay, so since the churned customers lead to revenue loss, I would want to focus on the, I want to predict all the customers that like predict the majority of the customers that have churned. So I would focus on recall in that case. Yeah. So that I don't want to incorrectly approve someone who has the chance of churning and then we do some loss. So yeah, I would prefer recall.",
            "feedback": "Prioritizing recall in a churn scenario often makes sense because you want to identify as many of the actual churners as possible so you can take action. But, the mention of 'incorrectly approved' could be refined, and tying the answer back to false negatives would make your reasoning stronger.",
            "score": 7.5,
            "improvement_areas": ["Explicitly talk about the cost of errors","Briefly mention the trade-off between precision and recall","Refine phrasing — avoid vague terms like 'incorrectly approve'"],
            "type": "technical",
            "followup_to": 8
        },
        {
            "id": 10,
            "question": "And, and you also talked about a trade-off between precision and recall. So are you aware about something called as an F1 score?",
            "answer": "Yeah. So again, because in this scenario, you said that churning customers have a more impact on the loss side, but if at all both are important, then if I need to look at the precision and the recall, then I would go with the F1 score, which it looks at the harmonic mean between the two.",
            "feedback": "You were right about the F1 score being the harmonic mean of precision and recall. That said, you could explain why harmonic mean (not arithmetic mean) is used (it penalizes extreme imbalances).",
            "score": 8.5,
            "improvement_areas": ["Briefly explain why harmonic mean is used (it penalizes extreme imbalances)","Provide a real-world use case where F1 is preferred, for example multi-class classification"],
            "type": "technical",
            "followup_to": 9
        },
        {
            "id": 11,
            "question": "Got it. And can you tell me some basic difference between like a linear model, for example, a logistic regression and tree-based models like XGBoost or random forest?",
            "answer": "So linear models are very standardized and they divide the, your data into, it's a very hard distinction, but with decision trees, at least it can learn the features on its own. It looks at all the features.",
            "feedback": "It's a good starting point, but several important aspects like linearity assumptions, interpretability, handling of feature interactions, and model flexibility were missing.",
            "score": 5,
            "improvement_areas": ["Explain that logistic regression assumes a linear relationship between features and the log-odds","Mention that tree-based models capture non-linear relationships and feature interactions automatically","Talk about higher interpretability of linear models"],
            "type": "technical",
            "followup_to": 8
        },
        {
            "id": 12,
            "question": "Okay. Fair enough. And for, so let's say you have XGBoost and you have random forest as tree-based models and logistic as a linear model. There is like a complexity and performance trade-off, which we call the bias variance trade-off. Are you aware about that? Can you just explain me what that is?",
            "answer": "Yes. So bias variance trade-off is again, like you said, it's about the complexity and performance of the model. Now, if the model complexity is too less, then your model would underfit. That means that it would not even, it would not perform well on your training data. And obviously not on your testing as well. Then there's the effect of overfitting where your model learns more than it needs to learn. It learns the noise of the noise in between the data and it will overfit, meaning that it will perform very well on the training data set, but will not perform that well on the test data set on unseen data points. So we don't want that to happen. We want it to be somewhere in the middle ground where it performs well on the unseen data as well. So that is what we call as a bias variance trade-off. And if it underfits, you can always increase the model complexity, but if it overfits, then you have to either decrease the model complexity or apply some regularization techniques or try to do some more cross-validation techniques, apply some cross-validation techniques or some other factors of that sort. And coming back to your question about linear regression and decision trees. So in terms of complexity, the decision trees are obviously complex, but like I said, because decision trees read more into the features. We want to see if it's overfitting or not. If it's not overfitting, then well, you can always go with decision trees, but if it does, then maybe sticking to a simpler module could be the optimal choice.",
            "feedback": "You did a good job capturing the essence of the bias-variance trade-off, particularly how model complexity affects underfitting (high bias) and overfitting (high variance). You also connected it well to practical model choices and mentioned techniques like regularization and cross-validation, which is excellent. YOu can think of providing slightly better definitions of bias and viariance.",
            "score": 9,
            "improvement_areas": ["Clearly define bias and variance","Briefly link linear models = high bias, low variance vs. tree-based = low bias, high variance","Avoid circling back too much; keep the answer more linear and crisp"],
            "type": "technical",
            "followup_to": 11
        },
        {
            "id": 13,
            "question": "Great. Yeah, fair enough. Okay. I think that is all the questions that I had for the technical side. Just a couple more questions, you know, on your experiences, on your most recent experiences, if you could share, let's start with, you know, an experience of you working in a team and let's say, I'm assuming, let's say you had some, if you had some conflicts with some team members or like if you wanted to put forward a point, but like the other team members were not in favor of that, or there was some conflict of ideas between team members. If you've ever come across that situation, how did you sort of work around that or work towards that?",
            "answer": "Yeah, so this happens in every team because that is how it functions in corporate. So yeah, it happened with me as well. It was during the first quarter of my program and it was in my leadership course. So our team, we were working on the, on one, on a Verizon churn case prediction model and like between the group and itself, we had two different opinions. I and one of my friends, we weren't in favor of using a different model, but the others strongly favored with a different model. So the one thing that I had that time that we realized was that the importance of a project manager. So we just, we put forward all of our points and we tried to convince the other team members why we were preparing this, like giving the reasoning of the complexity, the time left for the project and how well the model was performing. And then we just discussed it. And at the end, when everybody was convinced and on board, then that will help the option that I and my friend recommended.",
            "feedback": "You should add more clarity on your personal role, specific communication techniques you used, and what you learned from the experience. Give more detail about navigating the disagreement, and how you or someone else stepped in as the project manager.",
            "score": 7,
            "improvement_areas": ["Be clearer about your individual contribution to resolving the conflict","Highlight any specific soft skills used","More elaboration on the convincing process"],
            "type": "behavioral",
            "followup_to": null
        },
        {
            "id": 14,
            "question": "Nice. And what has been something very interesting that you've learned, I would say in your master's program so far. Technical or non-technical anything.",
            "answer": "Yeah. So very recently, like since data science is not my major, and this is the first time I'm dealing with all data science stuff. So I was very much impressed with neural networks, how they function and the large language models that everybody is using and relying so heavily upon. So I was flabbergasted with how it works and the functioning of all of it. And I'm kind of proud that I know how it works at the back end. And I'm not just a front end user of the LLM. So that is something about the technical side that I really liked too much on the non-technical part. Since most of my professors are already working in the industry. So I love the part that I get to learn the experiences. And since many of them work at very big firms like Amazon and Google, I get to know how they deal with their clients, how the storytelling works and the importance of how you should understand the problem statement. And it's not always the technical thing that matters, but also the non-technical stuff that takes you a long way.",
            "feedback": "The answer is genuine, enthusiastic, and provides a great balance between a specific technical learning and broader non-technical insights.",
            "score": 9,
            "improvement_areas": ["Share a specific LLM concept or project that deepened your understanding","Mention how you applied either of these learnings (in a project, class, etc.)"],
            "type": "behavioral",
            "followup_to": null
        }
    ],
    "overall_feedback": {
        "overall_score": 6.8,
        "scores": {
            "technical knowledge": 8,
            "communication"      : 8,
            "problem solving"    : 7.5,
            "behavioral"         : 8.5
        },
        "strengths": [
            "Demonstrates strong foundational and articulation understanding of ML, clustering, evaluation metrics, and hypothesis testing",
            "Shows genuine interest in learning, especially in new areas like LLMs",
            "Practical Application Focus",
            "Strong Project Storytelling",
            "Behavioral responses reflect team orientation and stakeholder awareness"
        ],
        "areas_of_improvement": [
            "Improve technical precision and structure in explanations (e.g., bias-variance, precision-recall)",
            "Use formal definitions and formulas where relevant to add depth",
            "Emphasizing Personal Contribution and growth in Behavioral Answers",
            "Avoid filler phrases; aim for concise, confident delivery"
        ]
    },
    "candidate_feedback": {
        "open_feedback": "I think the questions were good enough. You started with the project and then you drilled down and asked me more about the project in a sense. So that was good. Also for ML, you asked me about my comfort zone and then according to you just shifted through the topics and asked me something I don't really have any feedback. The questions were good. And the overall flow, like taking one topic and then diving deep into that. And then once we are done with once that once we have like a mutual satisfaction, then moving on to like different topics, it was a good approach. Amazing."
    }
}"""


def map_score_to_label(score: float) -> str:
    """Map a numeric score in [0, 5] to a qualitative label."""
    if score <= 1.0:
        return "Poor"
    if score <= 2.0:
        return "Fair"
    if score <= 3.0:
        return "Good"
    if score <= 4.0:
        return "Very Good"
    return "Excellent"


def aggregate_weighted_score(subscores: Dict[str, float]) -> float:
    """Compute the weighted aggregate score bounded to [0, 5]."""
    total = 0.0
    for key, weight in RUBRIC_WEIGHTS.items():
        value = float(subscores.get(key, 0.0))
        total += value * weight
    return max(0.0, min(5.0, total))


def default_heuristic_score(answer_text: str) -> Dict[str, float]:
    """Fallback heuristic scoring when no LLM is available."""
    text = (answer_text or "").strip()
    length = len(text.split())

    tech_keywords = [
        "model",
        "algorithm",
        "feature",
        "loss",
        "evaluate",
        "accuracy",
        "precision",
        "recall",
        "cluster",
        "k-means",
        "xgboost",
    ]
    tech_score = min(5.0, sum(1 for k in tech_keywords if k in text.lower()) * 1.2 + (0.01 * length))

    accuracy_score = 3.0 if length > 15 else 2.0 if length > 5 else 1.0

    hedges = ["maybe", "might", "could", "sort of", "i think", "probably", "not sure"]
    hedge_count = sum(1 for h in hedges if h in text.lower())
    confidence_score = max(1.0, 5.0 - hedge_count)

    step_words = ["first", "then", "next", "finally", "step", "approach", "process", "pipeline"]
    ps_score = min(5.0, sum(1 for w in step_words if w in text.lower()) * 1.3 + (0.005 * length))

    disfluencies = ["um", "uh", "ah", "erm"]
    disf_count = sum(1 for d in disfluencies if d in text.lower())
    flow_score = max(1.0, min(5.0, (length / 50.0) * 5.0 - disf_count))

    return {
        "technical_reasoning": round(tech_score, 2),
        "accuracy": round(accuracy_score, 2),
        "confidence": round(confidence_score, 2),
        "problem_solving": round(ps_score, 2),
        "flow": round(flow_score, 2),
    }


def _fallback_technical_feedback(subscores: Dict[str, float]) -> str:
    weakest = min(subscores.items(), key=lambda entry: entry[1])[0]
    mapping = {
        "technical_reasoning": (
            "Dive deeper into the core mechanics and trade-offs. Reference concrete models,"
            " data treatments, or evaluation math to strengthen the narrative."
        ),
        "accuracy": (
            "Ground the answer with verifiable metrics, benchmarks, or validation steps to"
            " demonstrate rigor."
        ),
        "confidence": (
            "Deliver conclusions decisively and reduce hedging; highlight prior successes to"
            " anchor your stance."
        ),
        "problem_solving": (
            "Lay out the end-to-end plan explicitly—define the problem, outline alternatives,"
            " and justify the chosen path."
        ),
        "flow": (
            "Tighten the storytelling arc. Remove filler words and use transitions to keep the"
            " interviewer oriented."
        ),
    }
    return mapping.get(weakest, "Provide more detail where the interviewer is probing.")


def _fallback_example_answer(question: str) -> str:
    prompt = question.strip()
    if not prompt:
        return (
            "A strong response should frame the challenge, detail the technical approach,"
            " highlight tooling, and finish with measurable impact."
        )
    return (
        "A high-scoring answer would: 1) frame the business or research context for the question"
        "; 2) outline the analytical or modeling approach with tooling choices; 3) discuss key"
        " metrics and validation; and 4) finish with measurable results or lessons learned."
    )


def _extract_json_candidate(text: str) -> Optional[Dict[str, float]]:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None


def _score_and_feedback_with_llm(question: str, answer: str) -> Optional[Dict[str, Any]]:
    if CLIENT is None:
        return None

    prompt_system = (
        "You are an expert interviewer evaluator for senior data science roles. For the provided"
        " question and answer you must return strict JSON with this schema: {\n"
        "  \"scores\": {\n"
        "    \"technical_reasoning\": float 0-5,\n"
        "    \"accuracy\": float 0-5,\n"
        "    \"confidence\": float 0-5,\n"
        "    \"problem_solving\": float 0-5,\n"
        "    \"flow\": float 0-5\n"
        "  },\n"
        "  \"technical_feedback\": string (two to three sentences highlighting strengths and actionable improvements),\n"
        "  \"example_answer\": string (a high-quality sample answer with clear structure and technical detail)\n"
        "}.\n"
        "Ensure the JSON is valid, without markdown fences, and keep the example answer grounded in the question.\n\n"
        "Reference evaluation for tone and structure guidance:\n"
    ) + REFERENCE_EVALUATION_EXAMPLE
    prompt_user = f"Question: {question}\nAnswer: {answer}"

    try:
        response = CLIENT.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user},
            ],
            temperature=0.0,
            max_tokens=400,
        )
    except Exception:
        return None

    message = response.choices[0].message.content if response.choices else ""
    parsed = _extract_json_candidate(message)
    if not parsed:
        return None

    scores = parsed.get("scores", parsed)
    result: Dict[str, Any] = {
        "scores": {
            "technical_reasoning": float(
                scores.get("technical_reasoning", scores.get("Technical Reasoning", 0.0))
            ),
            "accuracy": float(scores.get("accuracy", scores.get("Accuracy", 0.0))),
            "confidence": float(scores.get("confidence", scores.get("Confidence", 0.0))),
            "problem_solving": float(
                scores.get("problem_solving", scores.get("Problem-Solving", 0.0))
            ),
            "flow": float(scores.get("flow", scores.get("Flow", 0.0))),
        },
        "technical_feedback": parsed.get("technical_feedback")
        or parsed.get("Technical Feedback"),
        "example_answer": parsed.get("example_answer")
        or parsed.get("Example Answer"),
    }

    return result


def _summarize_session_with_llm(
    results: List[Dict[str, Any]],
    candidate_name: Optional[str],
) -> Optional[Dict[str, Any]]:
    if CLIENT is None or not results:
        return None

    payload = {
        "candidate_name": candidate_name or "the candidate",
        "responses": [
            {
                "question": item.get("question"),
                "answer": item.get("answer"),
                "final_score": item.get("final_score"),
                "rating": item.get("rating"),
                "technical_feedback": item.get("technical_feedback"),
            }
            for item in results
        ],
    }

    prompt_system = (
        "You are an AI interview coach. Given scored interview responses, craft a concise"
        " final evaluation: summarize overall performance with a constructive, strengths-first"
        " tone and enumerate concrete focus areas for improvement. Respond with strict JSON"
        " containing keys 'summary' (2-3 sentences) and 'areas_for_improvement' (array of 2-4"
        " short bullet-worthy strings).\n\n"
        "Reference evaluation for target voice and specificity:\n"
    ) + REFERENCE_EVALUATION_EXAMPLE
    prompt_user = json.dumps(payload)

    try:
        response = CLIENT.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user},
            ],
            temperature=0.3,
            max_tokens=400,
        )
    except Exception:
        return None

    message = response.choices[0].message.content if response.choices else ""
    parsed = _extract_json_candidate(message)
    if not parsed:
        return None

    areas = parsed.get("areas_for_improvement") or parsed.get("AreasForImprovement")
    if isinstance(areas, str):
        areas = [areas]

    return {
        "summary": parsed.get("summary") or parsed.get("Summary"),
        "areas_for_improvement": areas or [],
    }


def _compile_overall_feedback(
    results: List[Dict[str, Any]],
    candidate_name: Optional[str],
    use_llm: bool,
) -> Dict[str, Any]:
    if use_llm:
        llm_summary = _summarize_session_with_llm(results, candidate_name)
        if llm_summary:
            return llm_summary

    if not results:
        return {
            "summary": "No evaluative questions recorded; unable to compute overall feedback.",
            "areas_for_improvement": [],
        }

    avg_score = sum(item.get("final_score", 0.0) for item in results) / len(results)
    strongest = max(results, key=lambda item: item.get("final_score", 0.0))
    weakest = min(results, key=lambda item: item.get("final_score", 0.0))

    summary = (
        f"Overall performance is rated {map_score_to_label(avg_score)} with an average score of"
        f" {avg_score:.2f}. Maintain the strengths shown on '{strongest.get('question', 'key topics')}'."
    )
    areas = [
        f"Revisit '{weakest.get('question', 'weaker topics')}' incorporating: {weakest.get('technical_feedback')}"
    ]

    return {
        "summary": summary,
        "areas_for_improvement": areas,
    }


def evaluate_interview_json(interview: Dict, use_llm: bool = True) -> Dict:
    """Evaluate an interview transcript represented as a JSON-like dict."""
    questions: List[Dict] = interview.get("questions", [])
    icebreaker_count = int(interview.get("icebreaker_count", 0))
    results: List[Dict] = []

    for index, item in enumerate(questions, start=1):
        qid = item.get("id", index)
        question = item.get("question", "")
        answer = item.get("answer", "")
        is_icebreaker = bool(item.get("is_icebreaker")) or (index <= icebreaker_count)

        if is_icebreaker:
            results.append(
                {
                    "id": qid,
                    "question": question,
                    "answer": answer,
                    "skipped": True,
                    "skip_reason": SKIP_NOTE,
                    "technical_feedback": SKIP_NOTE,
                    "example_answer": None,
                    "subscores": {},
                    "final_score": None,
                    "rating": None,
                    "feedback": SKIP_NOTE,
                    "suggested_improvement": None,
                }
            )
            continue

        subscores = default_heuristic_score(answer)
        technical_feedback = _fallback_technical_feedback(subscores)
        example_answer = _fallback_example_answer(question)
        feedback_note = "Heuristic scoring used (no API key or use_llm=False)."

        if use_llm and CLIENT is not None:
            llm_result = _score_and_feedback_with_llm(question, answer)
            if llm_result:
                llm_scores = llm_result.get("scores", {})
                subscores = {
                    key: round(float(value), 2)
                    for key, value in llm_scores.items()
                }
                technical_feedback = llm_result.get("technical_feedback", technical_feedback)
                example_answer = llm_result.get("example_answer", example_answer)
                if isinstance(technical_feedback, list):
                    technical_feedback = " ".join(str(part) for part in technical_feedback)
                if isinstance(example_answer, list):
                    example_answer = " ".join(str(part) for part in example_answer)
                feedback_note = "LLM-assisted scoring applied."

        final_score = aggregate_weighted_score(subscores)
        rating = map_score_to_label(final_score)
        lowest_dimension = min(subscores.items(), key=lambda entry: entry[1])[0]
        recommendations = {
            "technical_reasoning": "Add more algorithmic detail, math, or code examples.",
            "accuracy": "Verify factual claims with sources or concrete numbers.",
            "confidence": "State conclusions decisively; reduce hedging language.",
            "problem_solving": "Outline a clear step-by-step approach or strategy.",
            "flow": "Reduce disfluencies and tighten the narrative structure.",
        }

        results.append(
            {
                "id": qid,
                "question": question,
                "answer": answer,
                "subscores": subscores,
                "final_score": round(final_score, 3),
                "rating": rating,
                "feedback": feedback_note,
                "suggested_improvement": recommendations[lowest_dimension],
                "technical_feedback": technical_feedback,
                "example_answer": example_answer,
                "skipped": False,
            }
        )

    evaluated_results = [item for item in results if not item.get("skipped")]
    overall_score = 0.0
    overall_rating = "Not Rated"
    if evaluated_results:
        overall_score = sum(item["final_score"] for item in evaluated_results) / len(evaluated_results)
        overall_rating = map_score_to_label(overall_score)

    overall_feedback = _compile_overall_feedback(
        evaluated_results,
        candidate_name=interview.get("candidate_name"),
        use_llm=use_llm,
    ) if evaluated_results else None

    return {
        "interview_id": interview.get("interview_id"),
        "candidate_name": interview.get("candidate_name"),
        "results": results,
        "overall_score": round(overall_score, 3) if evaluated_results else None,
        "overall_rating": overall_rating,
        "overall_feedback": overall_feedback,
    }


__all__ = [
    "evaluate_interview_json",
    "aggregate_weighted_score",
    "map_score_to_label",
    "default_heuristic_score",
    "RUBRIC_WEIGHTS",
]
