# Debugging Output Issues

## Why You See Zero Output

The system was running but output wasn't being displayed because:

1. **Result Format**: The `execution()` method returns a dictionary with keys like `"final_message"` and `"summary"`, but the code was checking `result.output` (attribute access) instead of `result.get("output")` (dictionary access).

2. **No Print Statements**: The output was only going to logs, not to stdout.

3. **Interrupted Execution**: You pressed Ctrl+C before the agent could complete and return results.

## Fixes Applied

1. ✅ Fixed prompt file paths to use absolute paths relative to script location
2. ✅ Fixed result access to use dictionary methods (`result.get("output")`)
3. ✅ Added print statements to display output to console
4. ✅ Added summary output display
5. ✅ Added better error messages when output is empty

## How to See Output Now

The system will now:
- Print output to console as it runs
- Display final results when complete
- Show summaries
- Log everything to files in `logs/` directory

## Running Again

```bash
cd /mnt/vdc/Para-Core/LLMPenPwn/agents
python3 linear_execution.py --subnet 10.10.110.0/24 --keys LLMPenPwn/agents/keys.cfg --json-output network_scan.json
```

The output will now be visible in the terminal as the agent runs.
