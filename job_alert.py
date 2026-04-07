import os, json, hashlib, requests
from jobspy import scrape_jobs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import *

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
SEEN_FILE = "seen_jobs.json"

# Load resume and vectorize
with open("resume.txt") as f:
    resume_text = f.read()

def score_job(description):
    """Cosine similarity between resume and job description."""
    if not description:
        return 0
    vec = TfidfVectorizer().fit_transform([resume_text, description])
    return round(cosine_similarity(vec[0], vec[1])[0][0] * 100, 1)

def count_skill_matches(description):
    desc_lower = description.lower()
    return sum(1 for s in SKILLS_REQUIRED if s.lower() in desc_lower)

def check_experience(description):
    """Returns True if experience requirement is within our range."""
    import re
    matches = re.findall(r'(\d+)\+?\s*(?:to\s*(\d+))?\s*years?', description.lower())
    for match in matches:
        low = int(match[0])
        high = int(match[1]) if match[1] else low
        if low <= EXPERIENCE_YEARS_MAX and high >= EXPERIENCE_YEARS_MIN:
            return True
    return not matches  # if no mention of years, don't filter out

def check_compensation(description):
    """Returns True if salary is acceptable or not mentioned."""
    import re
    # Look for LPA mentions
    matches = re.findall(r'(\d+)\s*(?:–|-|to)\s*(\d+)\s*lpa', description.lower())
    if matches:
        for low, high in matches:
            if int(high) >= COMPENSATION_MIN_LPA:
                return True
        return False
    return True  # no salary mentioned — don't filter out

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
    requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown"
    })

def main():
    seen = load_seen()

    jobs = scrape_jobs(
        site_name=["linkedin", "indeed"],
        search_term=" OR ".join(POSITIONS),
        location=", ".join(LOCATIONS),
        hours_old=24,
        results_wanted=50,
        country_indeed="India"
    )

    new_matches = 0

    for _, job in jobs.iterrows():
        title = str(job.get("title", ""))
        company = str(job.get("company", ""))
        url = str(job.get("job_url", ""))
        description = str(job.get("description", ""))
        salary = str(job.get("min_amount", "")) + "-" + str(job.get("max_amount", ""))

        # Deduplicate
        job_id = hashlib.md5(url.encode()).hexdigest()
        if job_id in seen:
            continue

        # Filter: company
        if COMPANIES and not any(c.lower() in company.lower() for c in COMPANIES):
            continue

        # Filter: skills
        if count_skill_matches(description) < SKILLS_MATCH_THRESHOLD:
            continue

        # Filter: experience
        if not check_experience(description):
            continue

        # Filter: compensation
        if not check_compensation(description):
            continue

        # Score against resume
        resume_score = score_job(description)
        if resume_score < 5:  # minimum similarity threshold
            continue

        # Send alert
        msg = (
            f"*New job match!* ({resume_score}% resume match)\n\n"
            f"*Role:* {title}\n"
            f"*Company:* {company}\n"
            f"*Skills matched:* {count_skill_matches(description)}/{len(SKILLS_REQUIRED)}\n"
            f"*Salary:* {salary if salary != 'None-None' else 'Not mentioned'}\n\n"
            f"{url}"
        )
        send_telegram(msg)
        seen.add(job_id)
        new_matches += 1

    save_seen(seen)
    print(f"Done. {new_matches} new matches sent.")

if __name__ == "__main__":
    main()