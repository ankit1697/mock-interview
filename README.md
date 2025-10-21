<h1 align="center">PRISM: Personal Response and Interview Simulation Model</h1>

<h3 align="center"> A product by Dialog</h3>

<p align="center">
  <img src="assets/logo1.png" width="400" alt="Logo">
</p>

## Project Motivation

> *“I read all the guides, practiced questions from LeetCode, and even watched mock interview videos, but the actual interview turned out to be very different.”*  
> *“I had practiced multiple technical questions. Yet, when the real interview started, I froze.”*  
> *“If only I could practice with an AI that actually understood my resume, asked realistic questions, and helped me improve.”*  


Jane is a student of Applied Data Science at the University of Chicago. She’s been on the job hunt for a couple of months now, applying to roles like *Data Scientist*, *Machine Learning Engineer*, and *Data Analyst* at her dream companies.

Jane is talented and hardworking, she’s comfortable with Python, knows how to tune a model, and can explain cross-validation without notes.  

But when it comes to **interviews**, she feels stuck.

She spends hours searching online for *“data science interview questions”*. Most results are generic or outdated, and she’s never sure which ones match the latest industry expectations. She occasionally practices with friends, but those mock interviews often turn into casual chats with little useful feedback. And while professional interview coaches promise tailored guidance, each session can cost **$100–$300**. Not something she can afford often as a student.

To fill the gap, Jane also turns to **AI tools like ChatGPT and Gemini** for help. They generate useful questions and explanations, but the experience feels **one-size-fits-all**. The AI doesn’t know her resume, her experience with time series forecasting, or the specific job she’s applying for. There’s no **context**, no **follow-up questioning**, and no **personalized feedback** on how she’s performing.

That’s the gap this **AI-powered mock interview platform** aims to fill. Helping users like Jane **practice smarter, not just harder**, through **context-aware, interactive, and adaptive AI interviews**.

---

### 🎯 The Problem

Preparing for technical interviews is often **overwhelming and inefficient**.  

- It takes around **30 hours to cover the basics** and up to **100 hours** to be well-prepared for technical interviews. [Tech Interview Handbook](https://www.techinterviewhandbook.org/coding-interview-prep/?utm_source=chatgpt.com)
- Even for non-technical interviews, candidates spend about **5–10 hours** on preparation. [Indeed](https://www.indeed.com/career-advice/interviewing/how-long-should-you-prepare-for-an-interview?utm_source=chatgpt.com)
- For DS/ML or coding roles, many candidates dedicate over **15 hours per week** leading up to interviews. [Helen Zhang, Medium](https://helen-zhang.medium.com/the-4-week-plan-to-nailing-your-next-coding-technical-interview-internship-level-c5368c47e1d?utm_source=chatgpt.com)
- Technical interviews themselves can last **30–60 minutes or more**, requiring on-the-spot reasoning and composure. [Coursera](https://www.coursera.org/articles/how-long-do-interviews-last?utm_source=chatgpt.com)
- Despite this effort, **65% of job seekers** say they struggle to find **trustworthy, realistic practice resources**. [Aptitude Research Report](https://www.aptituderesearch.com/wp-content/uploads/2022/06/Apt_Interviewing_Report-0622_Final.pdf?utm_source=chatgpt.com)
- Personalized coaching can cost **$100–$300 per session**, making consistent practice inaccessible. [Glassdoor & Coaching Industry Estimates](https://www.prnewswire.com/news-releases/glassdoor-study-reveals-interview-process-getting-longer-averaging-about-24-days-across-25-countries-300501746.html?utm_source=chatgpt.com)

The result?  
Even skilled candidates often **underperform** in interviews because they lack a safe, adaptive way to practice under realistic conditions.

---

### 💡 The Vision

We aim to build a platform where anyone can **practice, learn, and grow**.

- Upload your **resume and job description**. Input the role and industry. The AI instantly tailors questions to your experience.  
- The system **retrieves domain-specific technical questions** (via RAG from trusted data science sources).  
- Engage in a **dynamic, multi-turn conversation**, where follow-ups depend on your previous answers.  
- Receive **feedback on content, depth, communication, and confidence**, supported by curated corpus, interviews and use of vision models.  
- Track your **progress across sessions**, identifying improvement areas over time.

---

# Setting up django app

```
python3.9 -m venv venv
cd venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py runserver
```
