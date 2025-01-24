import sys
sys.path.append("../")
from scripts.alignment import align_answer,filter_answer
def test_alignment_functions():
    """
    Tests all alignment functions with various scenarios.
    """
    # Test Data
    test_cases = [
        {
            "description": "No policy violation, standard user",
            "input_text": "This is a general response about company policies.",
            "user_role": "employee",
            "expected_result": "This is a general response about company policies."
        },
        {
            "description": "Policy violation with restricted keyword",
            "input_text": "This response contains secret_code which should trigger a violation.",
            "user_role": "employee",
            "expected_result": "This answer is withheld due to policy restrictions."
        },
        {
            "description": "Policy violation with multiple keywords",
            "input_text": "This text includes both classified information and an internal_server.",
            "user_role": "admin",
            "expected_result": "This answer is withheld due to policy restrictions."
        },
        {
            "description": "No policy violation, manager role",
            "input_text": "Here are the results from last quarter's performance reviews.",
            "user_role": "manager",
            "expected_result": "Here are the results from last quarter's performance reviews."
        },
        {
            "description": "No policy violation, custom role",
            "input_text": "This is a response for a consultant.",
            "user_role": "consultant",
            "expected_result": "This is a response for a consultant."
        },
    ]

    # Test Execution
    for case in test_cases:
        print(f"Testing: {case['description']}")
        result = align_answer(case["input_text"], user_role=case["user_role"])
        assert result == case["expected_result"], f"Failed: {case['description']} | Expected: {case['expected_result']}, Got: {result}"
        print(f"Passed: {case['description']}\n")

if __name__ == "__main__":
    test_alignment_functions()


"""OUTPUT
Testing: No policy violation, standard user
Passed: No policy violation, standard user

Testing: Policy violation with restricted keyword
WARNING:scripts.alignment:Answer violates policy. Providing a restricted response.
Passed: Policy violation with restricted keyword

Testing: Policy violation with multiple keywords
WARNING:scripts.alignment:Answer violates policy. Providing a restricted response.
Passed: Policy violation with multiple keywords

Testing: No policy violation, manager role
Passed: No policy violation, manager role

"""