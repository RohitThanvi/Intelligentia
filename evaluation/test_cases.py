"""Example prompts, including the Section 30 demonstration scenario."""

LOAN_PROCESSING_SCENARIO = """\
A large financial institution processes approximately 25,000 loan applications
per month. Around 120 operations employees are involved in document
collection, verification, communication, exception handling, and application
processing. Processing currently takes 2-5 business days. Approximately 18%
of applications require additional communication because of missing or
inconsistent information.

The organization already has a CRM, loan-processing platform, document-
management system, workflow engine, and enterprise data warehouse.

Management is considering an AI/GenAI solution to reduce processing time,
improve employee productivity, automate document understanding and
communication, and improve customer experience.

Assume an average loaded employee cost of INR 7.5 lakh/year, an initial
implementation budget of INR 1.8 crore, expected annual operating cost of
INR 45 lakh, and a potential productivity improvement of approximately 30%.

Do not blindly accept these assumptions. Identify which assumptions require
validation.
"""

SIMPLE_SMOKE_TEST_SCENARIO = """\
We are a 40-person e-commerce company. Our customer support team of 5 people
handles about 200 emails a day, mostly order-status and return questions.
We want to know if GenAI could help, and roughly what it would cost and save.
"""

TEST_PROMPTS = [
    {"name": "loan_processing_full_scenario", "prompt": LOAN_PROCESSING_SCENARIO},
    {"name": "small_business_smoke_test", "prompt": SIMPLE_SMOKE_TEST_SCENARIO},
]
