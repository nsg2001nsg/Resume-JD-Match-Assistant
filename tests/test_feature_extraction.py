import unittest

from data_prep import (
    education_level_score,
    extract_education_level,
    extract_years,
    keyword_overlap_ratio,
    keyword_present,
)


class FeatureExtractionTests(unittest.TestCase):
    def test_extract_years_numeric_patterns(self):
        self.assertEqual(extract_years("Python developer with 5+ years of experience."), 5)
        self.assertEqual(extract_years("Experience of 7 years in accounting."), 7)
        self.assertEqual(extract_years("3 yrs exp with REST APIs."), 3)

    def test_extract_years_word_numbers(self):
        self.assertEqual(extract_years("Over five years of experience in backend development."), 5)
        self.assertEqual(extract_years("Minimum three years experience required."), 3)

    def test_extract_years_date_ranges(self):
        text = "Software Engineer, June 2019 - May 2024. Built Flask APIs and SQL reports."
        self.assertEqual(extract_years(text), 5)

    def test_extract_years_current_date_ranges(self):
        text = "Backend Developer Jan 2021 - Present. Built Python services."
        self.assertGreaterEqual(extract_years(text), 5)

    def test_extract_years_ignores_unrealistic_values(self):
        self.assertEqual(extract_years("Built 100 dashboards and 70 reports."), 0)

    def test_keyword_present_uses_token_boundaries(self):
        self.assertTrue(keyword_present("Built REST APIs with Python.", "REST API"))
        self.assertTrue(keyword_present("Experience with machine-learning pipelines.", "machine learning"))
        self.assertFalse(keyword_present("Used javascript and typescript.", "java"))

    def test_keyword_overlap_ratio_uses_phrase_matching(self):
        resume = "Built REST APIs with Python and PostgreSQL."
        keywords = ["rest api", "python", "java"]
        self.assertEqual(keyword_overlap_ratio(resume, keywords), 0.6667)

    def test_education_level_avoids_common_false_positives(self):
        self.assertEqual(extract_education_level("Worked with MS Office and cloud platforms."), 3)
        self.assertEqual(extract_education_level("Able to be productive in teams."), 3)

    def test_education_level_score(self):
        resume = "Education: MCA from Bangalore University."
        jd = "Requires bachelor's degree in computer science."
        self.assertEqual(education_level_score(resume, jd), 1)


if __name__ == "__main__":
    unittest.main()
