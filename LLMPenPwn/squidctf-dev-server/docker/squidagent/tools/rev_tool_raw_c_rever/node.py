from parser import CFunctionParser
import sys
import copy
import json
import logging
from openai import OpenAI

# Global logging flag
ENABLE_LOGGING = True

# Setup logging
if ENABLE_LOGGING:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ai_prompts.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

with open("keys.cfg", "r") as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("OPENAI_API_KEY="):
            OPENAI_API_KEY = line.strip().split("=")[1]
            break
client = OpenAI(api_key=OPENAI_API_KEY)

ALL_NODES=[]


class Node_parsers:
    def __init__(self, name, type, call_graph=None, function=None, raw_function=None, new_function=None, json_function=None, parent=None, children=None, data_ref=None, parser=None):
        self.name = name
        self.type = type
        self.parser = parser
        if type == "FUNCTION":
            # print(function)
            self.children = call_graph[name]
            self.function = function
            self.raw_function = raw_function
            self.new_function = new_function
            self.json_function = json_function
            self.weight = self.find_weight()
        if type == "DATA_REF":
            self.parent = data_ref.parents
            self.weight = self.find_weight()

    def find_weight(self):
        if self.type == "FUNCTION":
            if len(self.children) == 0:
                return 100000000
            else:
                cnt = 0
                for i in self.parser.global_vars:
                    if self.name in i.parents:
                        cnt+=1
                return -1 * self.parser.get_weight_children(self.function['name']) + 2 * self.parser.get_weight_parents(self.function['name']) - cnt
        b = sum(1 for i in self.parent) 
        return 100000000000 + b


def parser_all_functions(parser: CFunctionParser):
    for function in parser.functions:
        node = Node_parsers(type="FUNCTION", name=function['name'], call_graph=parser.call_graph, function=function, raw_function=function['full_function'], new_function=None, json_function=None, parser=parser)
        ALL_NODES.append(node)
    for data_ref in parser.global_vars:
        node = Node_parsers(type="DATA_REF", name=data_ref.name, data_ref=data_ref)
        ALL_NODES.append(node)

def create_queue(nodes):
    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[0]
        left = [x for x in arr[1:] if x.weight > pivot.weight]
        right = [x for x in arr[1:] if x.weight <= pivot.weight]
        return quicksort(left) + [pivot] + quicksort(right)
    return quicksort(list(nodes))


def call_all_AI(nodes):
    queue = create_queue(nodes)
    json_nodes = []
    for node in queue:
        if node.type == "FUNCTION":
            function_AI_call_clean(node)
            function_AI_call_json_rename(node)
            json_nodes.append(node)  # Add processed function node
        elif node.type == "DATA_REF":
            rename_data_ref(node)
            json_nodes.append(node)  # Add processed data ref node
    return json_nodes

def function_AI_call_clean(node):
    prompt = f"""
    Please clean the function: {node.name}
    The code should be cleaned up and the code should be in the same format as the original code.
    Do not remove any Variables or Functions or Constants or Structures or Enums or Macros or Types or Labels or Comments or Preprocessor Directives or other code elements.
    Do not remove any code that is not needed for the function to work.
    Your output should only be the cleaned up code that makes further analysis easier.
    
    NEVER SKIP THIS STEP, EVEN IF THE FUNCTION IS EMPTY, EVEN IF YOU CAN'T FIND A PERFECT NEW NAME PLEASE PICK A GOOD NAME, ADD THE COMMENT "End of Function: NEW_FUNCTION_NAME"
    At the end of the code, please add a line that says "End of Function: NEW_FUNCTION_NAME"

    The new function name should describe the function in a way that describes its functionality.
    It should be a short and concise name that describes the function in a way that is easy to understand.
    The new function name should be a name that is not already used in the code.

    Here is the Source Code of the function:
    {node.function['full_function']}
    """

    # Log the prompt if logging is enabled
    if ENABLE_LOGGING:  
        logger.info(f"=== FUNCTION CLEAN PROMPT for {node.name} ===")
        logger.info(f"System: You are a C based Program Analysis Assistant that is able to analyze the code and generate a clean up decomplied C code.")
        logger.info(f"User: {prompt}")
        logger.info("=" * 50)

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a C based Program Analysis Assistant that is able to analyze the code and generate a clean up decomplied C code."},
            {"role": "user", "content": prompt}, # function_AI_call_clean
        ]
    )
    code = response.choices[0].message.content
    
    # Log the response if logging is enabled
    if ENABLE_LOGGING:
        logger.info(f"=== FUNCTION CLEAN RESPONSE for {node.name} ===")
        logger.info(f"Response: {code}")
        logger.info("=" * 50)
    node.new_function = code.split("End of Function: ")[0]
    node.name = code.split("End of Function: ")[1].strip()
    for i in ALL_NODES:
        if i.type == "DATA_REF":
            if node.name in i.parent:
                for j in i.parent:
                    if j == node.name:
                        j = node.name



def function_AI_call_json_rename(node):

    prompt = f"""
    Please create a description that describes how the function: {node.name} works
    Please do not include the code just the description, 
    The description should:
    - be a description of the function in a way that is easy to understand.
    - only be one paragraph
    - include any key variables or functions or constants or structures or enums or macros or types or labels or comments or preprocessor directives or other code elements that are used in the function.
    - be in the same language as the function.
    - make sure to avoid any general descriptions, it should be specific to the function.
    - not make any assumptions, it should be based on the code.
    - double check the code and make sure to not miss any details.
    Here is the Source Code of the function:
    {node.function['full_function']}
    """
    
    # Log the prompt if logging is enabled
    if ENABLE_LOGGING:
        logger.info(f"=== FUNCTION JSON RENAME PROMPT for {node.name} ===")
        logger.info(f"System: You are a C based Program Analysis Assistant that is able to analyze the code and generate a clean up decomplied C code.")
        logger.info(f"User: {prompt}")
        logger.info("=" * 50)
    
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a C based Program Analysis Assistant that is able to analyze the code and generate a clean up decomplied C code."},
            {"role": "user", "content": prompt}, # function_AI_call_clean
        ]
    )
    node.json_function = response.choices[0].message.content
    
    # Log the response if logging is enabled
    if ENABLE_LOGGING:
        logger.info(f"=== FUNCTION JSON RENAME RESPONSE for {node.name} ===")
        logger.info(f"Response: {node.json_function}")
        logger.info("=" * 50)

def rename_data_ref(node):
    json_data = []
    for parent in node.parent:
        json = json_data_creat(parent, node)
        json_data.append([parent,json])
        json_data.append("\n")
    prompt = f"""
    Please generate a good name for this data_ref: {node.name}
    PLEASE USE CAMEL CASE FOR THE NEW NAME.
    ALWAYS RETURN THE NEW NAME, DO NOT RETURN ANYTHING ELSE. EVEN IF YOU CAN'T FIND A PERFECT NEW NAME PLEASE PICK A GOOD NAME.
    ALWAYS RETURN THE NEW NAME, DO NOT RETURN ANYTHING ELSE. EVEN IF YOU CAN'T FIND A PERFECT NEW NAME PLEASE PICK A GOOD NAME.
    ALWAYS RETURN THE NEW NAME, DO NOT RETURN ANYTHING ELSE. EVEN IF YOU CAN'T FIND A PERFECT NEW NAME PLEASE PICK A GOOD NAME.
    ALWAYS RETURN THE NEW NAME, DO NOT RETURN ANYTHING ELSE. EVEN IF YOU CAN'T FIND A PERFECT NEW NAME PLEASE PICK A GOOD NAME.
    The new data_ref name should describe the data_ref in a way that describes its functionality.
    It should be a short and concise name that describes the data_ref in a way that is easy to understand.
    The new data_ref name should be a name that is not already used in the code.
    Given the following json descriptions/data of the parent functions that uses the data_ref:
    {json_data}
    """

    # Log the prompt if logging is enabled
    if ENABLE_LOGGING:
        logger.info(f"=== DATA REF RENAME PROMPT for {node.name} ===")
        logger.info(f"System: You are a C based Program Analysis Assistant that is able to analyze the code and generate a clean up decomplied C code.")
        logger.info(f"User: {prompt}")
        logger.info("=" * 50)

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a C based Program Analysis Assistant that is able to analyze the code and generate a clean up decomplied C code."},
            {"role": "user", "content": prompt}, # function_AI_call_clean
        ]
    )
    a = response.choices[0].message.content
    
    # Log the response if logging is enabled
    if ENABLE_LOGGING:
        logger.info(f"=== DATA REF RENAME RESPONSE for {node.name} ===")
        logger.info(f"Response: {a}")
        logger.info("=" * 50)
    for parent in node.parent:
        for i in ALL_NODES:
            if i.type == "FUNCTION":
                if i.name == parent:
                    i.raw_function = i.raw_function.replace(node.name, a)
    node.name = a
        

def json_data_creat(parent, node):

    prompt = f"""
    Please create a description that describes how the function: {parent} uses the data_ref: {node.name}
    Please create a description that describes how the function: {parent} uses the data_ref: {node.name}
    Please create a description that describes how the function: {parent} uses the data_ref: {node.name}
    Please create a description that describes how the function: {parent} uses the data_ref: {node.name}
    The description should:
    - be a description of the data_ref in a way that is easy to understand.
    - only be one paragraph
    - include any key variables or functions or constants or structures or enums or macros or types or labels or comments or preprocessor directives or other code elements that are used in the data_ref.
    - be in the same language as the data_ref.
    - make sure to avoid any general descriptions, it should be specific to the data_ref.
    - not make any assumptions, it should be based on the code.
    - double check the code and make sure to not miss any details.
    Here is the Source Code of the function: 
    """

    for i in ALL_NODES:
        if i.type == "FUNCTION":
            if i.name == parent:
                prompt += "\n"
                prompt += i.raw_function # the code of the function
                if ENABLE_LOGGING:
                    logger.info(f"=== JSON DATA CREATE PROMPT for parent {parent} and data_ref {node.name} ===")
                    logger.info(f"System: You are a C based Program Analysis Assistant that is able to analyze the code and generate a clean up decomplied C code.")
                    logger.info(f"User: {prompt}")
                    logger.info("=" * 50)
                response = client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[
                        {"role": "system", "content": "You are a C based Program Analysis Assistant that is able to analyze the code and generate a clean up decomplied C code."},
                        {"role": "user", "content": prompt}, # function_AI_call_clean
                    ]
                )
                response_content = response.choices[0].message.content
                
                # Log the response if logging is enabled
                if ENABLE_LOGGING:
                    logger.info(f"=== JSON DATA CREATE RESPONSE for parent {parent} and data_ref {node.name} ===")
                    logger.info(f"Response: {response_content}")
                    logger.info("=" * 50)
                
                return response_content
    return None

def create_clean_code(json_nodes,filename):
    #print(f"Processing {len(json_nodes)} nodes for file: {filename}")
    
    with open(filename, "r") as old_code:
        old_code_lines = old_code.readlines()
    new_code_lines = copy.deepcopy(old_code_lines)
    
    for node in reversed(json_nodes):
        if node.type == "FUNCTION":
            #print(f"Processing function: {node.name}")
            #print(f"Function has new_function: {node.new_function is not None}")
            if node.new_function:
                start_index = node.function['start_line'] - 1  
                end_index = node.function['end_line'] - 1
                #print(f"Replacing lines {start_index+1} to {end_index+1}")
                del new_code_lines[start_index:end_index + 1]
                new_lines = node.new_function.split('\n')
                for i, line in enumerate(new_lines):
                    new_code_lines.insert(start_index + i, line + '\n')
    
    output_filename = filename.replace(".c", "_clean.c")
    print(f"Writing cleaned code to: {output_filename}")
    with open(output_filename, "w") as new_code:
        new_code.writelines(new_code_lines)
    print(f"Successfully wrote {len(new_code_lines)} lines to {output_filename}")

# testing version 2         
# def create_clean_code(json_nodes,filename):
#     print(f"Processing {len(json_nodes)} nodes for file: {filename}")
#     new_code_lines = []
#     for node in json_nodes:
#         if node.type == "FUNCTION":
#             new_code_lines.append('\n'+node.new_function + '\n')
#     output_filename = filename.replace(".c", "_clean.c")
#     print(f"Writing cleaned code to: {output_filename}")
#     with open(output_filename, "w") as new_code:
#         new_code.writelines(new_code_lines)
#     print(f"Successfully wrote {len(new_code_lines)} lines to {output_filename}")
            


def main():
    if len(sys.argv) != 2:
        print("to test: python node.py <c_file>")
        return
    filename = sys.argv[1]
    #print(f"Parsing C file: {filename}")
    parser = CFunctionParser(filename)
    parser_all_functions(parser)
    json_nodes = call_all_AI(ALL_NODES)
    create_clean_code(json_nodes,filename)

if __name__ == "__main__":
    main()