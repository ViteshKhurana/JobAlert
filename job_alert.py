import os, json, hashlib, requests
from jobspy import scrape_jobs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import *

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
SEEN_FILE = "seen_jobs.json"

# Load resume
with open("resume.txt") as f:
    resume_text = f.read()

def score_job(description):
    if not description:
        return 0
    vec = TfidfVectorizer().fit_transform([resume_text, description])
    return round(cosine_similarity(vec[0], vec[1])[0][0] * 100, 1)

def company_score(company):
    company_lower = company.lower()
    if any(c in company_lower for c in TOP_COMPANIES):
        return 20   # strong boost
    return 0

def count_skill_matches(description):
    desc_lower = description.lower()
    return sum(1 for s in SKILLS_REQUIRED if s in desc_lower)

def check_experience(description):
    import re
    matches = re.findall(r'(\d+)\+?\s*(?:to\s*(\d+))?\s*years?', description.lower())
    for match in matches:
        low = int(match[0])
        high = int(match[1]) if match[1] else low
        if low <= EXPERIENCE_YEARS_MAX and high >= EXPERIENCE_YEARS_MIN:
            return True
    return True  # neutral if not mentioned

def extract_salary_score(description):
    import re
    matches = re.findall(r'(\d+)\s*(?:–|-|to)\s*(\d+)\s*lpa', description.lower())
    if matches:
        for low, high in matches:
            if int(high) >= COMPENSATION_MIN_LPA:
                return 10   # good salary
            else:
                return -10  # low salary
    return 0  # neutral (no salary mentioned)

def load_seen():
    try:
        with open(SEEN_FILE) as f:
            return set(json.load(f))
    except:
        return set()

def save_seen(seen):
    with open(SEEN_FILE, "w") as f:
        json.dump(list(seen), f)

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    res = requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown"
    })
    print("Telegram:", res.status_code)

def main():
    print("Fetching jobs...")

    seen = load_seen()

    jobs = scrape_jobs(
        site_name=["linkedin", "indeed"],
        search_term=" OR ".join(POSITIONS),
        location=", ".join(LOCATIONS),
        hours_old=24,
        results_wanted=50,
        country_indeed="India"
    )

    print("Total jobs fetched:", len(jobs))

    candidates = []

    for _, job in jobs.iterrows():
        title = str(job.get("title", ""))
        company = str(job.get("company", ""))
        url = str(job.get("job_url", ""))
        description = str(job.get("description", ""))

        job_id = hashlib.md5(url.encode()).hexdigest()
        if job_id in seen:
            continue

        # --- Scoring ---
        score = 0

        skills_matched = count_skill_matches(description)
        # ❗ HARD FILTER: skills must match
        if skills_matched < MIN_SKILL_MATCH:
            continue
        score += skills_matched * 10

        resume_score = score_job(description)
        score += resume_score

        if check_experience(description):
            score += 10

        salary_score = extract_salary_score(description)
        score += salary_score

        # ✅ Company boost
        company_boost = company_score(company)
        score += company_boost

        print(f"{title} | Score: {score} | Skills: {skills_matched} | Resume: {resume_score}")

        # ❗ SOFT FILTER: overall quality
        if score < MIN_SCORE_THRESHOLD:
            continue

        candidates.append((score, job, job_id, skills_matched, resume_score))

    # Sort by best score
    candidates.sort(key=lambda x: x[0], reverse=True)

    sent = 0

    for score, job, job_id, skills_matched, resume_score in candidates[:5]:
        title = str(job.get("title", ""))
        company = str(job.get("company", ""))
        url = str(job.get("job_url", ""))

        msg = (
            f"*Top Job Match* 🚀\n\n"
            f"*Role:* {title}\n"
            f"*Company:* {company}\n"
            f"*Score:* {round(score,1)}\n"
            f"*Skills matched:* {skills_matched}/{len(SKILLS_REQUIRED)}\n"
            f"*Resume match:* {resume_score}%\n\n"
            f"{url}"
        )

        send_telegram(msg)
        seen.add(job_id)
        sent += 1

    save_seen(seen)

    print(f"Done. {sent} jobs sent.")

if __name__ == "__main__":
    main()