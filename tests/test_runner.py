import os, json, tempfile
from mock_interview.runner import run_full_pass

def test_run_full_pass_mock(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_MODE", "1")
    inp = tmp_path/"in.json"
    inp.write_text(json.dumps({"questions":[{"id":"q1","question":"LR?","answer":"Sigmoid","type":"technical"}]}))
    outp = tmp_path/"out.json"
    res = run_full_pass(str(inp), str(outp))
    assert outp.exists()
    assert res["meta"]["graded_questions"] == 1
