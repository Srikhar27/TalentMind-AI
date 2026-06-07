import re
from datetime import datetime

class CandidateRankingEngine:
    def __init__(self, jd_requirements):
        self.jd = jd_requirements
        # Weights for the scoring framework
        self.weights = {
            "semantic": 0.35,
            "skill": 0.20,
            "experience": 0.15,
            "project": 0.10,
            "behavior": 0.10,
            "education": 0.05,
            "quality": 0.05
        }
        self.current_date = datetime(2026, 6, 7) # current date of the challenge

    def parse_date(self, date_str):
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except:
            return None

    def calculate_job_months(self, start_s, end_s):
        start_d = self.parse_date(start_s)
        end_d = self.parse_date(end_s) if end_s else self.current_date
        if start_d and end_d:
            return (end_d.year - start_d.year) * 12 + (end_d.month - start_d.month)
        return 0

    def is_honeypot(self, cand):
        """
        Identifies honeypot trap candidates with subtly impossible profiles.
        """
        history = cand.get('career_history', [])
        skills = cand.get('skills', [])
        education = cand.get('education', [])
        
        # 1. Job duration mismatch: e.g. duration_months is much greater than actual date span
        for job in history:
            start_s = job.get('start_date')
            end_s = job.get('end_date')
            duration = job.get('duration_months', 0)
            
            calc_months = self.calculate_job_months(start_s, end_s)
            if calc_months > 0 and (duration - calc_months) > 12:
                return True
                
        # 2. Expert/advanced skills with 0 duration (>= 5 skills)
        expert_zero = sum(1 for s in skills if s.get('proficiency') in ['expert', 'advanced'] and s.get('duration_months', 0) == 0)
        if expert_zero >= 5:
            return True
            
        # 3. Education start year > end year
        for edu in education:
            sy = edu.get('start_year')
            ey = edu.get('end_year')
            if sy and ey and sy > ey:
                return True
                
        # 4. Job start date > end date
        for job in history:
            start_s = job.get('start_date')
            end_s = job.get('end_date')
            start_d = self.parse_date(start_s)
            end_d = self.parse_date(end_s) if end_s else None
            if start_d and end_d and start_d > end_d:
                return True
                
        return False

    def compute_semantic_score(self, sim_score):
        """
        Normalize FAISS cosine similarity to [0, 100].
        Cosine similarity for match is typically in range [0.3, 0.9].
        """
        # Linear normalization: map [0.3, 0.85] to [0, 100]
        min_sim, max_sim = 0.30, 0.85
        score = (sim_score - min_sim) / (max_sim - min_sim) * 100
        return max(0.0, min(100.0, score))

    def compute_skill_score(self, candidate_skills):
        """
        Skill Match Score = Matched Skills / Required Skills (proficiency-weighted)
        """
        required = self.jd["required_skills"]
        preferred = self.jd["preferred_skills"]
        
        cand_skills = {s.get('name', '').lower().strip(): s for s in candidate_skills}
        
        prof_weights = {
            "expert": 1.0,
            "advanced": 0.85,
            "intermediate": 0.65,
            "beginner": 0.35
        }
        
        matched_score = 0.0
        # Required skills are weighted higher (1.0 each)
        for req in required:
            matched = False
            for k, v in cand_skills.items():
                if req in k or k in req: # Substring overlap
                    prof = v.get('proficiency', 'intermediate').lower()
                    matched_score += prof_weights.get(prof, 0.65) * 1.0
                    matched = True
                    break
                    
        # Preferred skills are weighted lower (0.5 each)
        for pref in preferred:
            matched = False
            for k, v in cand_skills.items():
                if pref in k or k in pref:
                    prof = v.get('proficiency', 'intermediate').lower()
                    matched_score += prof_weights.get(prof, 0.65) * 0.5
                    matched = True
                    break

        # Max possible score
        max_possible = len(required) * 1.0 + len(preferred) * 0.5
        score = (matched_score / max_possible) * 100
        return max(0.0, min(100.0, score))

    def compute_experience_score(self, cand):
        """
        Evaluates years of experience, gaps, career progression, consulting firm check, and job-hopping.
        """
        profile = cand.get('profile', {})
        history = cand.get('career_history', [])
        
        # 1. Total years of experience (ideal range 5-9, perfect is 6-8)
        years = profile.get('years_of_experience', 0.0)
        exp_score = 100.0
        if years < 5.0:
            # Linear penalty for juniority: 20 points per year below 5
            exp_score -= (5.0 - years) * 20
        elif years > 9.0:
            # Gradual penalty for over-qualification: 5 points per year above 9
            exp_score -= (years - 9.0) * 5
            
        exp_score = max(0.0, exp_score)
        
        # 2. Employment gaps (penalize if gap between consecutive jobs is > 6 months)
        gap_penalty = 0.0
        parsed_jobs = []
        for job in history:
            s_d = self.parse_date(job.get('start_date'))
            e_d = self.parse_date(job.get('end_date'))
            if s_d:
                parsed_jobs.append((s_d, e_d if e_d else self.current_date))
                
        # Sort jobs by start date descending
        parsed_jobs.sort(key=lambda x: x[0], reverse=True)
        for i in range(len(parsed_jobs) - 1):
            curr_start = parsed_jobs[i][0]
            prev_end = parsed_jobs[i+1][1]
            if curr_start > prev_end:
                gap_months = (curr_start.year - prev_end.year) * 12 + (curr_start.month - prev_end.month)
                if gap_months > 6:
                    gap_penalty += 10.0 # Deduct 10 points per large gap
                    
        # 3. Consulting companies check (TCS, Infosys, Wipro, etc. - JD mentions "bad fit experiences")
        consulting_count = 0
        for job in history:
            comp = job.get('company', '').lower().strip()
            if any(cc in comp for cc in self.jd["disqualifiers"]["companies"]):
                consulting_count += 1
                
        # Penalize if their entire career is in consulting, or if the current role is consulting
        consulting_penalty = 0.0
        if len(history) > 0:
            if consulting_count == len(history):
                consulting_penalty = 25.0
            elif consulting_count > 0:
                consulting_penalty = 10.0 * (consulting_count / len(history))

        # 4. Job hopping check (average tenure < 18 months)
        hopping_penalty = 0.0
        total_tenure_months = sum(job.get('duration_months', 0) for job in history)
        if len(history) > 1:
            avg_tenure = total_tenure_months / len(history)
            if avg_tenure < 18.0:
                # Up to 20 points penalty for frequent switching
                hopping_penalty = (18.0 - avg_tenure) / 18.0 * 20.0
                
        final_exp_score = exp_score - gap_penalty - consulting_penalty - hopping_penalty
        return max(0.0, min(100.0, final_exp_score))

    def compute_project_score(self, history):
        """
        Evaluate project description relevance to key JD words (search, retrieval, embeddings, etc.).
        """
        project_keywords = [
            "embedding", "retrieval", "vector", "search", "ranking", "faiss", 
            "nlp", "re-rank", "evaluation", "ndcg", "mrr", "map", "lora", 
            "recommendation", "matching", "pipeline", "fine-tuning"
        ]
        
        matches = 0
        total_desc_len = 0
        
        for job in history:
            desc = job.get('description', '').lower()
            total_desc_len += len(desc)
            for kw in project_keywords:
                if kw in desc:
                    matches += 1
                    
        if total_desc_len == 0:
            return 0.0
            
        # Score based on matching keywords (log-scaled frequency)
        score = (matches / len(project_keywords)) * 100
        # If they match at least 4 key concepts, we scale it to a solid score
        score = score * 2.0
        return max(0.0, min(100.0, score))

    def compute_behavioral_score(self, signals):
        """
        Compute score from redrob_signals (last_active, recruiter response, avg response time, etc.).
        """
        score = 0.0
        total_weight = 0.0
        
        # 1. Recruiter Response Rate (weight 2.0)
        rrr = signals.get('recruiter_response_rate', 0.0)
        score += rrr * 100 * 2.0
        total_weight += 2.0
        
        # 2. Avg Response Time Hours (weight 1.5)
        # Target is < 24 hours. > 168 hours (1 week) gets 0.
        resp_time = signals.get('avg_response_time_hours', 168.0)
        if resp_time <= 24.0:
            score += 100 * 1.5
        elif resp_time <= 48.0:
            score += 80 * 1.5
        elif resp_time <= 72.0:
            score += 60 * 1.5
        elif resp_time <= 168.0:
            score += (168.0 - resp_time) / (168.0 - 72.0) * 40 * 1.5
        total_weight += 1.5
        
        # 3. Interview Completion Rate (weight 2.0)
        icr = signals.get('interview_completion_rate', 0.0)
        score += icr * 100 * 2.0
        total_weight += 2.0
        
        # 4. Offer Acceptance Rate (weight 1.0)
        oar = signals.get('offer_acceptance_rate', -1)
        if oar >= 0.0:
            score += oar * 100 * 1.0
            total_weight += 1.0
            
        # 5. Last Active Date Recency (weight 1.5)
        last_active_s = signals.get('last_active_date')
        last_active_d = self.parse_date(last_active_s)
        if last_active_d:
            days_inactive = (self.current_date - last_active_d).days
            if days_inactive <= 30:
                score += 100 * 1.5
            elif days_inactive <= 90:
                score += 80 * 1.5
            elif days_inactive <= 180:
                score += 50 * 1.5
            else:
                score += max(0.0, 50 - (days_inactive - 180) / 10) * 1.5
        total_weight += 1.5
        
        # 6. Open To Work Flag (weight 1.0)
        otw = signals.get('open_to_work_flag', False)
        score += (100 if otw else 50) * 1.0
        total_weight += 1.0
        
        # 7. Github Activity Score (weight 1.0)
        gh = signals.get('github_activity_score', -1)
        if gh >= 0:
            score += gh * 1.0
            total_weight += 1.0
        else:
            score += 30 * 1.0 # default penalty for no github
            total_weight += 1.0
            
        # 8. Profile Completeness Score (weight 1.0)
        pc = signals.get('profile_completeness_score', 0.0)
        score += pc * 1.0
        total_weight += 1.0
        
        return score / total_weight

    def compute_education_score(self, education):
        """
        Degree relevance and college tiering (tier_1/tier_2).
        """
        if not education:
            return 30.0 # baseline
            
        best_edu_score = 0.0
        
        tier_scores = {
            "tier_1": 100.0,
            "tier_2": 80.0,
            "tier_3": 50.0,
            "tier_4": 30.0
        }
        
        for edu in education:
            tier = edu.get('tier', 'tier_4').lower().strip()
            tier_score = tier_scores.get(tier, 30.0)
            
            deg = str(edu.get('degree', '')).lower()
            field = str(edu.get('field_of_study', '')).lower()
            
            # Relevance weight
            relevance = 0.4
            if any(tf in field for tf in ['computer science', 'information technology', 'software engineering', 'data science', 'artificial intelligence', 'machine learning', 'electronics']):
                relevance = 1.0
            elif any(tf in field for tf in ['engineering', 'mathematics', 'statistics', 'physics']):
                relevance = 0.8
            elif any(tf in field for tf in ['commerce', 'business', 'mba', 'management']):
                relevance = 0.5
                
            edu_val = tier_score * relevance
            if edu_val > best_edu_score:
                best_edu_score = edu_val
                
        return max(30.0, min(100.0, best_edu_score))

    def compute_profile_quality_score(self, cand):
        """
        Detect keyword stuffing, empty profiles, salary anomalies, and experience inconsistencies.
        """
        profile = cand.get('profile', {})
        history = cand.get('career_history', [])
        skills = cand.get('skills', [])
        signals = cand.get('redrob_signals', {})
        education = cand.get('education', [])
        
        score = 100.0
        
        # 1. Keyword stuffing
        summary = profile.get('summary', '').lower()
        headline = profile.get('headline', '').lower()
        words = summary.split()
        if len(words) > 0:
            word_counts = {}
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1
            max_word_freq = max(word_counts.values()) if word_counts else 0
            if max_word_freq > 15 and max_word_freq / len(words) > 0.15:
                score -= 30.0 # keyword stuffing penalty
                
        # 2. Empty profile
        if not history and not skills:
            return 0.0
            
        # 3. Inconsistencies
        # Expected salary min > max
        salary = signals.get('expected_salary_range_inr_lpa', {})
        if salary and 'min' in salary and 'max' in salary and salary['min'] > salary['max']:
            score -= 40.0
            
        # Job duration mismatch (flagged as quality penalty if not filtered out, though we filter honeypots)
        has_job_mismatch = False
        for job in history:
            start_s = job.get('start_date')
            end_s = job.get('end_date')
            duration = job.get('duration_months', 0)
            calc_months = self.calculate_job_months(start_s, end_s)
            if calc_months > 0 and (duration - calc_months) > 12:
                has_job_mismatch = True
                score -= 50.0
                break
                
        # Skill duration exceeds total experience + 12 months
        years_exp = profile.get('years_of_experience', 0.0)
        for s in skills:
            if s.get('duration_months', 0) > (years_exp * 12 + 12):
                score -= 20.0
                break
                
        # Education start year > end year
        for edu in education:
            sy = edu.get('start_year')
            ey = edu.get('end_year')
            if sy and ey and sy > ey:
                score -= 50.0
                break
                
        return max(0.0, min(100.0, score))

    def score_candidate(self, cand, sim_score):
        """
        Calculates all sub-scores and final weighted score.
        Returns dictionary of scores.
        """
        is_hp = self.is_honeypot(cand)
        
        # Calculate sub-scores
        semantic = self.compute_semantic_score(sim_score)
        skill = self.compute_skill_score(cand.get('skills', []))
        experience = self.compute_experience_score(cand)
        project = self.compute_project_score(cand.get('career_history', []))
        behavior = self.compute_behavioral_score(cand.get('redrob_signals', {}))
        education = self.compute_education_score(cand.get('education', []))
        quality = self.compute_profile_quality_score(cand)
        
        # If honeypot, quality goes to 0 and we force final score to be very low
        if is_hp:
            quality = 0.0
            final = 0.1 # Floor score for honeypots so they rank at the absolute bottom
        else:
            final = (
                self.weights["semantic"] * semantic +
                self.weights["skill"] * skill +
                self.weights["experience"] * experience +
                self.weights["project"] * project +
                self.weights["behavior"] * behavior +
                self.weights["education"] * education +
                self.weights["quality"] * quality
            )
            
        return {
            "candidate_id": cand.get('candidate_id'),
            "semantic_score": round(semantic, 2),
            "skill_score": round(skill, 2),
            "experience_score": round(experience, 2),
            "project_score": round(project, 2),
            "behavior_score": round(behavior, 2),
            "education_score": round(education, 2),
            "quality_score": round(quality, 2),
            "final_score": round(final, 2),
            "is_honeypot": is_hp
        }
