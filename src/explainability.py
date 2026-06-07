class CandidateExplainability:
    def __init__(self, jd_requirements):
        self.jd = jd_requirements

    def generate_reason(self, cand, score_details):
        """
        Generates a custom 1-2 sentence explanation of why the candidate is ranked where they are.
        Uses specific profile facts (title, experience, skills, location, signals) and connects to JD.
        """
        profile = cand.get('profile', {})
        skills = cand.get('skills', [])
        signals = cand.get('redrob_signals', {})
        
        name = profile.get('anonymized_name', 'The candidate')
        title = profile.get('current_title', 'Engineer')
        exp = profile.get('years_of_experience', 0.0)
        
        # Extract candidate key skills
        cand_skills = [s.get('name', '') for s in skills]
        
        # Identify matched core required skills
        matched_required = [req for req in self.jd["required_skills"] if any(req in s.lower() for s in cand_skills)]
        
        # Find key highlights
        highlights = []
        if matched_required:
            highlights.append(f"strong hands-on experience in {', '.join(matched_required[:3])}")
        
        # Location matching
        location = profile.get('location', '').lower()
        loc_match = any(loc in location for loc in self.jd["locations"])
        
        # Github activity
        gh_score = signals.get('github_activity_score', -1)
        
        # Recruiter response rate
        rrr = signals.get('recruiter_response_rate', 0.0)
        
        # Select reasoning templates based on final score tier
        final_score = score_details.get('final_score', 0.0)
        
        if score_details.get('is_honeypot'):
            return f"Profile contains severe timeline and skill metadata inconsistencies, which flag it as a non-genuine application."

        # Case 1: High rank (Top candidates)
        if final_score >= 80.0:
            part1 = f"{name} is an excellent fit as a {title} with {exp:.1f} years of experience, demonstrating "
            if highlights:
                part1 += f"{highlights[0]}."
            else:
                part1 += "strong alignment with the ML/AI engineering requirements."
                
            part2 = ""
            if loc_match:
                part2 = f" They are based in {profile.get('location')}, offering immediate local availability."
            elif gh_score > 50:
                part2 = f" Their active GitHub presence (score: {gh_score:.1f}) highlights practical code contributions."
            elif rrr > 0.7:
                part2 = f" Excellent recruiter response rate ({rrr*100:.0f}%) indicates high availability and interest."
                
            return (part1 + part2).strip()

        # Case 2: Mid rank (Good fit but with some gaps)
        elif final_score >= 60.0:
            part1 = f"Strong profile as a {title} with {exp:.1f} years of experience and core skills in "
            if matched_required:
                part1 += f"{', '.join(matched_required[:2])}."
            else:
                part1 += "machine learning."
                
            # Document minor gaps/concerns
            gaps = []
            if exp < 5.0:
                gaps.append("has slightly less than the desired 5 years of experience")
            elif exp > 9.0:
                gaps.append("is somewhat over-qualified for this Series A role")
                
            if rrr < 0.4:
                gaps.append("shows lower responsiveness to recruiters (response rate: {rrr*100:.0f}%)")
            if signals.get('notice_period_days', 0) > 90:
                gaps.append("has a long notice period of {signals.get('notice_period_days')} days")
                
            if gaps:
                part2 = f" However, the candidate {gaps[0]}."
            else:
                part2 = " Gaps in hybrid search or vector retrieval experience prevent a higher ranking."
                
            return (part1 + part2).strip()

        # Case 3: Low rank (Poor fit / irrelevant)
        else:
            gaps = []
            if not matched_required:
                gaps.append("lacks relevant embeddings/retrieval skills")
            if exp < 3.0:
                gaps.append(f"has limited experience ({exp:.1f} years)")
            elif exp > 12.0:
                gaps.append(f"is highly senior ({exp:.1f} years)")
                
            reason = f"Ranked lower because the candidate's profile as a {title} "
            if gaps:
                reason += f"{' and '.join(gaps[:2])}."
            else:
                reason += "does not align with the intelligence layer architecture requirements."
                
            return reason
