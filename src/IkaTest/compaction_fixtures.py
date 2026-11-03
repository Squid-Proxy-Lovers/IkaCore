"""Deterministic synthetic histories shared by offline and live compaction evaluation."""

import json


def retention_fixture(index=0):
    facts = {"goal": f"Map archive sector {index + 17}", "constraint": "Never modify source files; budget 37 credits",
             "result": f"Archived probe returned marker AMBER-{index + 300}", "question": "Who approves the final transfer?",
             "pending": "Await checksum for parcel Q9"}
    call = {"role": "assistant", "content": None, "reasoning_content": "Check integrity before transfer",
            "tool_calls": [{"id": "lookup", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]}
    result = {"role": "tool", "tool_call_id": "lookup", "content": "Newest tool confirmed ready"}
    pending = {"role": "assistant", "content": None, "tool_calls": [{"id": "pending", "type": "function",
               "function": {"name": "checksum", "arguments": json.dumps({"parcel": "Q9"})}}]}
    history = {"system": {"message": facts["constraint"], "tokens": 20},
               "first_input": {"message": facts["goal"], "tokens": 10}, "summary": {"message": "", "tokens": 0},
               "messages": {"noise-" + str(i): {"message": "Archived observation " + str(i) + " " + "filler " * 20, "tokens": 30} for i in range(80)}}
    history["messages"]["noise-13"]["message"] = facts["result"]
    history["messages"]["noise-41"]["message"] = facts["question"]
    history["messages"].update({
        "user": {"type": "hitl_input", "message": "Continue and preserve prior open questions.", "tokens": 10},
        "call": {"type": "assistant_with_tools", "message": json.dumps(call), "reasoning_content": "Check integrity before transfer", "tokens": 20},
        "result": {"type": "tool", "message": json.dumps(result), "tokens": 20},
        "pending": {"type": "assistant_with_tools", "message": json.dumps(pending), "tokens": 20},
        "pending-note": {"message": facts["pending"], "tokens": 10},
    })
    return facts, history

