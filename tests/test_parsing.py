from mock_interview.loaders import normalize_from_questions_block

def test_normalize_from_questions_block():
    data = {"questions":[{"id":"q1","question":"What?","answer":"This","type":"technical"}]}
    items = normalize_from_questions_block(data)
    assert items[0]["id"] == "q1"
    assert items[0]["answer"] == "This"
