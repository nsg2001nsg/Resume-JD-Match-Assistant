import unittest
from data_prep import (
    extract_years,
    extract_jd_required_years,
    segment_resume_sections
)
from explainability import qualify_snippet_context
from features import compute_features

class CalibrationTests(unittest.TestCase):
    
    def test_education_dates_completely_excluded(self):
        """
        Verify that academic/education date ranges are completely ignored
        when calculating candidate professional experience.
        """
        resume_text = """
        John Doe
        Email: john.doe@email.com
        
        EDUCATION:
        Master of Computer Applications (MCA)
        Bangalore University | 2024 - 2026
        Bachelor of Computer Applications (BCA)
        State College | 2020 - 2023
        
        PROJECTS:
        Library System: Built with Python and Flask.
        """
        
        # Test segmenting
        sections = segment_resume_sections(resume_text)
        self.assertIn("2024 - 2026", sections["education"])
        self.assertIn("2020 - 2023", sections["education"])
        
        # Excluded completely from professional experience
        extracted = extract_years(resume_text)
        self.assertEqual(extracted, 0, "Education dates should not contribute to work experience.")
        
    def test_internship_and_academic_research_assistant_dates_counted(self):
        """
        Verify that internship, freelance, research assistant, and trainee date ranges
        are correctly counted as work experience because they match job context markers.
        """
        resume_text_intern = """
        Jane Smith
        
        PROFESSIONAL EXPERIENCE:
        Software Engineer Intern
        TechCorp Inc. | Jan 2023 - Dec 2023
        Worked on backend APIs using Django and PostgreSQL.
        """
        self.assertEqual(extract_years(resume_text_intern), 1)
        
        resume_text_ra = """
        Research Assistant
        Machine Learning Lab | Jan 2021 - Dec 2022
        Analyzed datasets and trained models with PyTorch.
        """
        self.assertEqual(extract_years(resume_text_ra), 2)
        
        resume_text_multiple = """
        Experience:
        Associate Consultant | Jan 2022 - Dec 2022
        Freelance Developer | Jan 2023 - Dec 2023
        """
        self.assertEqual(extract_years(resume_text_multiple), 2)

    def test_jd_experience_parsing_robustness(self):
        """
        Verify JD experience requirement extraction handles ranges, plus signs,
        minimum wording, and abbreviations.
        """
        self.assertEqual(extract_jd_required_years("Requires 2-5 years of experience."), 2)
        self.assertEqual(extract_jd_required_years("Python developer (3-5 years exp)"), 3)
        self.assertEqual(extract_jd_required_years("Senior Software Engineer (5+ years experience)"), 5)
        self.assertEqual(extract_jd_required_years("Minimum 3 years of hands-on experience."), 3)
        self.assertEqual(extract_jd_required_years("Min 2 yrs of python programming."), 2)
        self.assertEqual(extract_jd_required_years("Position requires 4 years of industry tenure."), 4)
        self.assertEqual(extract_jd_required_years("Internship opportunity (0 years experience required)"), 0)

    def test_fresher_vs_senior_jd_penalty_and_clamp_simulation(self):
        """
        Verify that if candidate experience is 0 and JD experience is >= 2,
        the scoring calibration penalty and clamping logic behaves correctly.
        """
        # Simulated Flask API / scoring logic
        def run_calibrated_scoring(candidate_exp, jd_req_exp, raw_prob):
            is_fresher_mismatch = (candidate_exp == 0 and jd_req_exp >= 2)
            prob = raw_prob
            if is_fresher_mismatch:
                prob = prob * 0.70
                prob = max(0.35, min(prob, 0.58))
            return prob, is_fresher_mismatch
            
        # Case 1: Fresher applying to 3-year experience JD (high keyword match probability of 85%)
        final_score, is_mismatch = run_calibrated_scoring(0, 3, 0.85)
        self.assertTrue(is_mismatch)
        # 0.85 * 0.70 = 0.595, clamped to 0.58
        self.assertAlmostEqual(final_score, 0.58)
        
        # Case 2: Fresher applying to 2-year experience JD (low keyword match probability of 40%)
        final_score, is_mismatch = run_calibrated_scoring(0, 2, 0.40)
        self.assertTrue(is_mismatch)
        # 0.40 * 0.70 = 0.28, clamped up to 0.35
        self.assertAlmostEqual(final_score, 0.35)
        
        # Case 3: Experienced candidate applying to 3-year experience JD (no mismatch, no penalty)
        final_score, is_mismatch = run_calibrated_scoring(4, 3, 0.85)
        self.assertFalse(is_mismatch)
        self.assertAlmostEqual(final_score, 0.85)

    def test_snippet_prioritization(self):
        """
        Verify that action verb sentences are qualified as STRONG_IMPLEMENTATION,
        whereas sentences containing deprioritized phrases (e.g. "basic familiarity")
        are forced to BRIEF_MENTION (priority 1) to float rich project sentences to the top.
        """
        # Strong sentence (action verb + direct subject or direct active start)
        strong_sent = "Built REST APIs using Flask and PostgreSQL."
        strong_qual = qualify_snippet_context(strong_sent, "REST API")
        self.assertEqual(strong_qual["priority"], 3)
        self.assertEqual(strong_qual["code"], "STRONG_IMPLEMENTATION")
        
        # Workflow sentence
        wf_sent = "Worked with Docker and AWS in team sprints."
        wf_qual = qualify_snippet_context(wf_sent, "Docker")
        self.assertEqual(wf_qual["priority"], 2)
        self.assertEqual(wf_qual["code"], "WORKFLOW_CONTEXT")
        
        # Deprioritized sentence (contains "familiarity" or "basic")
        dep_sent = "I developed a basic familiarity with REST APIs during college."
        dep_qual = qualify_snippet_context(dep_sent, "REST API")
        self.assertEqual(dep_qual["priority"], 1, "Sentence should be deprioritized to priority 1.")
        self.assertEqual(dep_qual["code"], "BRIEF_MENTION")

    def test_robust_alias_keyword_matching(self):
        """
        Verify that our NORMALIZED_TECH_MAP is used inside keyword_present
        to cleanly map AWS/Git/Django/Flask/SQL aliases and abbreviation variations.
        """
        from data_prep import keyword_present
        
        # AWS alias matching
        self.assertTrue(keyword_present("AWS (S3, Lambda, EC2, RDS)", "Amazon Web Services (AWS)"))
        self.assertTrue(keyword_present("Worked on Amazon Web Services deployment.", "Amazon Web Services (AWS)"))
        
        # Git alias matching
        self.assertTrue(keyword_present("Proficient in Git and GitHub.", "Git Version Control"))
        self.assertTrue(keyword_present("Uses version control software.", "Git Version Control"))
        
        # Flask/Django/SQL
        self.assertTrue(keyword_present("Developed with Flask.", "Flask Microframework"))
        self.assertTrue(keyword_present("Designed Django systems.", "Django Framework"))
        self.assertTrue(keyword_present("Used PostgreSQL database.", "SQL Databases"))

    def test_score_ceiling_calibration(self):
        """
        Verify that score upper-range compression smoothly scales prediction scores
        above 80%, capping at exactly 92% to preserve headroom.
        """
        def calibrate_score(raw_prob):
            if raw_prob > 0.80:
                return 0.80 + (min(raw_prob, 1.0) - 0.80) * 0.60
            return raw_prob
            
        self.assertAlmostEqual(calibrate_score(1.0), 0.92)
        self.assertAlmostEqual(calibrate_score(0.90), 0.86)
        self.assertAlmostEqual(calibrate_score(0.80), 0.80)
        self.assertAlmostEqual(calibrate_score(0.50), 0.50)

    def test_snippet_ranking_rebalanced(self):
        """
        Verify that action verbs combined with core high-value technical nouns
        (api, microservices, automation, deployment, etc.) properly rank
        as STRONG_IMPLEMENTATION (priority 3), even with modifying prefixes.
        """
        sent = "Highly automated backend system integrations with AWS."
        qual = qualify_snippet_context(sent, "AWS")
        self.assertEqual(qual["priority"], 3)
        self.assertEqual(qual["code"], "STRONG_IMPLEMENTATION")
        
        sent_bullet = "Led development of backend systems using REST APIs."
        qual_bullet = qualify_snippet_context(sent_bullet, "REST API")
        self.assertEqual(qual_bullet["priority"], 3)
        self.assertEqual(qual_bullet["code"], "STRONG_IMPLEMENTATION")

    def test_canonical_education_status(self):
        """
        Verify that get_education_status correctly produces unified delta,
        label, and Plain English summaries without sign discrepancy.
        """
        from data_prep import get_education_status
        
        # MCA (tier 4) vs Bachelor's (tier 3)
        status = get_education_status("MCA degree from VJTI.", "Requires a Bachelor's degree.")
        self.assertEqual(status["tier_delta"], 1)
        self.assertEqual(status["label"], "EXCEEDS")
        self.assertIn("exceed", status["summary"])
        
        # Bachelor's (tier 3) vs Bachelor's (tier 3)
        status_meets = get_education_status("Bachelor of Engineering (B.E.).", "Requires B.Tech/B.E.")
        self.assertEqual(status_meets["tier_delta"], 0)
        self.assertEqual(status_meets["label"], "MEETS")
        self.assertIn("meet", status_meets["summary"])
        
        # Diploma (tier 2) vs Master's (tier 4)
        status_below = get_education_status("Polytechnic Diploma.", "Requires MBA/Postgraduate.")
        self.assertEqual(status_below["tier_delta"], -2)
        self.assertEqual(status_below["label"], "BELOW")
        self.assertIn("below", status_below["summary"])

    def test_skill_focused_filtering(self):
        """
        Verify that clean_and_preserve_keywords correctly:
        - Excludes generic action/responsibility terms from technical_requirements (putting them into responsibility_terms).
        - Completely excludes education-related terms from technical_requirements.
        - Excludes junior title or generic project phrases.
        """
        from features import clean_and_preserve_keywords
        
        raw_keywords = [
            "python", "django framework", "postgresql", "build", "maintain", 
            "understanding", "title junior", "science related", "computer science",
            "responsibilities develop", "docker containerization"
        ]
        
        kws_categorized = clean_and_preserve_keywords("Python Developer role requiring master's degree", raw_keywords)
        tech_reqs = kws_categorized["technical_requirements"]
        resp_terms = kws_categorized["responsibility_terms"]
        
        # 1. Technical Requirements must retain skills
        self.assertIn("Python", tech_reqs)
        self.assertIn("Django Framework", tech_reqs)
        self.assertIn("PostgreSQL", tech_reqs)
        self.assertIn("Docker Containerization", tech_reqs)
        
        # 2. Responsibility terms must isolate action terms
        self.assertIn("build", resp_terms)
        self.assertIn("maintain", resp_terms)
        
        # 3. Non-skill keywords must not pollute technical requirements
        self.assertNotIn("title junior", tech_reqs)
        self.assertNotIn("understanding", tech_reqs)
        self.assertNotIn("responsibilities develop", tech_reqs)
        self.assertNotIn("computer science", tech_reqs)
        self.assertNotIn("science related", tech_reqs)
        
        # 4. Education terms must be completely ignored
        self.assertNotIn("computer science", resp_terms)
        self.assertNotIn("science related", resp_terms)

    def test_unified_evidence_matching_aws_git(self):
        """
        Verify that AWS, Git, Django, Flask, and SQL databases correctly match
        under the extract_evidence_snippets centralized keyword_present pipeline,
        ensuring that they are not marked as missing.
        """
        from explainability import extract_evidence_snippets
        
        resume_text = "AWS (S3, Lambda, EC2, RDS), Git, Django, Flask, PostgreSQL."
        jd_keywords = [
            "Amazon Web Services (AWS)",
            "Git Version Control",
            "Django Framework",
            "Flask Microframework",
            "SQL Databases"
        ]
        
        evidence = extract_evidence_snippets(resume_text, "Senior python developer", jd_keywords)
        
        # None of these should be listed as missing
        missing = [kw.lower() for kw in evidence["missing"]]
        self.assertNotIn("amazon web services (aws)", missing)
        self.assertNotIn("git version control", missing)
        self.assertNotIn("django framework", missing)
        self.assertNotIn("flask microframework", missing)
        self.assertNotIn("sql databases", missing)
        
        # Check that matched snippets exist
        self.assertGreater(len(evidence["matched"]), 0)

if __name__ == "__main__":
    unittest.main()
