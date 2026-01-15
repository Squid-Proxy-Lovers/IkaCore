import json
import re
import sys
file_path = sys.argv[1]
output_file_path = sys.argv[2]
with open(file_path, 'r') as file:
    log_data = json.load(file)

    #print(f"[{entry['role']}]")
transcript = log_data['transcript']

regex = r'Tool call: \{.*?name.*?:.*?agent'
agent_calls = []
for i in range(len(transcript)):
    entry = transcript[i]
    if (entry['role'] == 'console'):
        if re.search(regex, entry['content'],re.DOTALL):
            if i < len(transcript) - 1:
                print(entry['content'])
                agent_calls.append(entry['content'])
                print("\n---\n")
with open(output_file_path, 'w') as output_file: 
    for i in range(len(agent_calls)):
        raw_text = agent_calls[i]
        output_file.write("\n\n\n--- Agent Call ---\n\n\n")
        start_index = raw_text.find('{')
        if start_index != -1:
            clean_json = raw_text[start_index:]
        else:
            print("No JSON object found in string")
            continue

        try:
            entry = json.loads(clean_json)
            output_file.write(entry['name'] + "\n")
            output_file.write(entry['arguments']['payload'] + "\n")
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")