import re
import sys
import os

class Data_ref:
    def __init__(self, name, type, value):
        self.name = name
        self.type = type
        self.value = value
        self.parents = []


class CFunctionParser:
    def __init__(self, filename):
        self.filename = filename
        self.functions = []
        self.call_graph = {}
        self.global_vars = []
        self.find_functions()
        self.generate_call_graph()
        self.find_global_variables()
        self.find_global_variable_usage()
        
    def remove_comments_and_strings(self, code):
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', code)
        code = re.sub(r"'(?:[^'\\]|\\.)*'", "''", code)
        return code
    
    def find_functions(self):
        try:
            with open(self.filename, 'r', encoding='utf-8') as file:
                content = file.read()
        except FileNotFoundError:
            print(f"Error: File '{self.filename}' not found.")
            return []
        except Exception as e:
            print(f"Error reading file: {e}")
            return []
        
        clean_content = self.remove_comments_and_strings(content)
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if (not line or 
                line.startswith('#') or 
                line.startswith('//') or 
                line.startswith('/*') or
                line.startswith('*') or
                line.startswith('typedef') or
                line.startswith('struct ') or
                line.startswith('enum ') or
                line.startswith('union ') or
                line.startswith('//-----') or
                line.startswith('// Data declarations') or
                line.startswith('// Function declarations')):
                i += 1
                continue
            
            potential_func_lines = []
            j = i
            brace_found = False
            while j < len(lines) and j < i + 5:
                potential_func_lines.append(lines[j])
                if '{' in lines[j]:
                    brace_found = True
                    break
                j += 1
            
            if not brace_found:
                i += 1
                continue
            
            combined = ' '.join(line.strip() for line in potential_func_lines)
            
            if self.is_potential_function(combined):
                func_start_line = i
                brace_line = j
                end_line = self.find_function_end(lines, brace_line)
                if end_line is not None:
                    func_info = self.extract_function_info(lines, func_start_line, end_line)
                    if func_info and func_info['name'] != '__int64':  # Skip incorrectly parsed global vars
                        self.functions.append(func_info)
                i = end_line + 1 if end_line else j + 1
            else:
                i += 1
        return self.functions
    
    def is_potential_function(self, text):
        if '(' not in text or ')' not in text or '{' not in text:
            return False
        before_brace = text[:text.find('{')].strip()
        if before_brace.endswith(';'):
            return False
        if not re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)', before_brace):
            return False
        control_keywords = ['if', 'while', 'for', 'switch', 'sizeof', 'typeof', 'return']
        for keyword in control_keywords:
            if re.search(r'\b' + keyword + r'\s*\(', text):
                return False
        return True
    
    def find_function_end(self, lines, start_line):
        brace_count = 0
        for i in range(start_line, len(lines)):
            line = lines[i]
            for char in line:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return i
        return None
    
    def extract_function_info(self, lines, start_line, end_line):
        func_lines = lines[start_line:end_line + 1]
        full_function = '\n'.join(func_lines)
        signature_lines = []
        for line in func_lines:
            signature_lines.append(line)
            if '{' in line:
                brace_pos = line.find('{')
                if brace_pos > 0:
                    signature_lines[-1] = line[:brace_pos]
                elif brace_pos == 0:
                    signature_lines.pop()
                break
        signature = ' '.join(line.strip() for line in signature_lines).strip()
        func_name = self.extract_function_name(signature)
        if not func_name:
            return None
        return_type, params = self.parse_signature(signature, func_name)
        return {
            'name': func_name,
            'signature': signature,
            'full_function': full_function.strip(),
            'start_line': start_line + 1,
            'end_line': end_line + 1,
            'return_type': return_type,
            'params': params
        }
    
    def extract_function_name(self, signature):
        match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', signature)
        if match:
            potential_name = match.group(1)
            keywords = ['if', 'while', 'for', 'switch', 'sizeof', 'typeof', 'return', 
                       'static', 'extern', 'inline', 'const', 'volatile', 'unsigned', 
                       'signed', 'void', 'int', 'char', 'short', 'long', 'float', 
                       'double', 'struct', 'enum', 'union']
            if potential_name not in keywords:
                return potential_name
        return None
    
    def parse_signature(self, signature, func_name):
        name_pos = signature.find(func_name)
        if name_pos == -1:
            return "void", []
        return_type = signature[:name_pos].strip()
        if not return_type:
            return_type = "void"
        match = re.search(r'\((.*?)\)', signature[name_pos:])
        if not match:
            return return_type, []
        params_str = match.group(1).strip()
        if not params_str or params_str == "void":
            return return_type, []
        params = []
        param_parts = self.split_params(params_str)
        for i, param in enumerate(param_parts):
            param = param.strip()
            if not param:
                continue
            param_info = self.parse_parameter(param, i)
            if param_info:
                params.append(param_info)
        return return_type, params
    
    def split_params(self, params_str):
        params = []
        current = ""
        paren_count = 0
        for char in params_str:
            if char == ',' and paren_count == 0:
                params.append(current.strip())
                current = ""
            else:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                current += char
        if current.strip():
            params.append(current.strip())
        return params
    
    def parse_parameter(self, param, index):
        if '(*' in param or '( *' in param:
            match = re.search(r'\(\s*\*\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)', param)
            if match:
                name = match.group(1)
                return {'name': name, 'type': param}
            else:
                return {'name': f'param{index}', 'type': param}
        array_suffix = ""
        if '[' in param:
            array_match = re.search(r'(\[[^\]]*\])', param)
            if array_match:
                array_suffix = array_match.group(1)
                param = param[:param.find('[')]
        tokens = re.split(r'(\s+|\*+)', param)
        tokens = [t for t in tokens if t and not t.isspace()]
        if not tokens:
            return None
        name = None
        type_tokens = []
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] != '*' and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tokens[i]):
                type_keywords = ['void', 'int', 'char', 'short', 'long', 'float', 
                               'double', 'struct', 'enum', 'union', 'const', 
                               'volatile', 'unsigned', 'signed', 'static', 
                               'extern', 'register', 'auto']
                if tokens[i] not in type_keywords or i == len(tokens) - 1:
                    name = tokens[i]
                    type_tokens = tokens[:i] + tokens[i+1:]
                    break
        if not name:
            name = f'param{index}'
            type_tokens = tokens
        type_str = ""
        for token in type_tokens:
            if token == '*':
                type_str += token
            else:
                if type_str and not type_str.endswith('*'):
                    type_str += " "
                type_str += token
        type_str = type_str.strip() + array_suffix
        if not type_str:
            type_str = "int"
        return {'name': name, 'type': type_str}
    
    def generate_call_graph(self):
        for func in self.functions:
            self.call_graph[func['name']] = []
            func_body = func['full_function']
            for other_func in self.functions:
                if other_func['name'] == func['name']:
                    continue
                pattern = r'(?<!\w)' + re.escape(other_func['name']) + r'\s*\('
                if re.search(pattern, func_body):
                    self.call_graph[func['name']].append(other_func['name'])
        return self.call_graph
    
    def find_global_variables(self):
        try:
            with open(self.filename, 'r', encoding='utf-8') as file:
                content = file.read()
        except:
            return []
        
        clean_content = self.remove_comments_and_strings(content)
        lines = clean_content.split('\n')
        
        function_ranges = []
        for func in self.functions:
            function_ranges.append((func['start_line'] - 1, func['end_line'] - 1))
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if self.is_inside_function(i, function_ranges):
                i += 1
                continue
                
            if (not line or 
                line.startswith('#') or 
                line.startswith('//') or 
                line.startswith('/*') or
                line.startswith('*') or
                line.startswith('typedef') or
                line.startswith('struct ') or
                line.startswith('enum ') or
                line.startswith('union ') or
                line.endswith('{') or
                line.endswith('}')):
                i += 1
                continue
            
            declaration_lines = []
            j = i
            while j < len(lines):
                declaration_lines.append(lines[j].strip())
                if ';' in lines[j]:
                    break
                if '{' in lines[j] or '}' in lines[j]:
                    break
                j += 1
            
            if j < len(lines) and ';' in lines[j]:
                combined_declaration = ' '.join(declaration_lines)
                if self.is_global_declaration(combined_declaration):
                    var_info = self.parse_global_declaration(combined_declaration, i + 1)
                    if var_info:
                        for var in var_info:
                            data_ref = Data_ref(var['name'], var['type'], var['init_value'])
                            data_ref.initialized = var['initialized']
                            data_ref.init_value = var['init_value']
                            data_ref.line_number = var['line_number']
                            self.global_vars.append(data_ref)
            
            i = j + 1
        
        return self.global_vars
    
    def is_inside_function(self, line_num, function_ranges):
        for start, end in function_ranges:
            if start <= line_num <= end:
                return True
        return False
    
    def is_global_declaration(self, declaration):
        declaration = declaration.strip()
        if not declaration or not declaration.endswith(';'):
            return False
        
        if '(' in declaration and ')' in declaration:
            return False
        
        if declaration.startswith('#') or declaration.startswith('typedef'):
            return False
        
        if re.search(r'\b(struct|enum|union)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\{', declaration):
            return False
        
        if re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\s*(\[[^\]]*\])?\s*(=|;)', declaration):
            return True
        
        return False
    
    def parse_global_declaration(self, declaration, line_num):
        declaration = declaration.rstrip(';').strip()
        variables = []
        
        storage_classes = ['static', 'extern', 'register', 'auto']
        type_qualifiers = ['const', 'volatile', 'restrict']
        base_types = ['void', 'char', 'short', 'int', 'long', 'float', 'double', 'signed', 'unsigned', '_UNKNOWN', '__int64']
        
        tokens = re.split(r'(\s+|\*+|,)', declaration)
        tokens = [t for t in tokens if t and not t.isspace()]
        
        type_part = []
        var_part = []
        collecting_type = True
        
        for i, token in enumerate(tokens):
            if token == ',':
                if var_part:
                    var_info = self.parse_single_variable(' '.join(type_part), ' '.join(var_part), line_num)
                    if var_info:
                        variables.append(var_info)
                    var_part = []
                    collecting_type = False
            elif collecting_type and (token in storage_classes or token in type_qualifiers or token in base_types or token.startswith('struct') or token.startswith('enum') or token.startswith('union')):
                type_part.append(token)
            elif token == '*':
                if collecting_type:
                    type_part.append(token)
                else:
                    var_part.append(token)
            else:
                collecting_type = False
                var_part.append(token)
        
        if var_part:
            var_info = self.parse_single_variable(' '.join(type_part), ' '.join(var_part), line_num)
            if var_info:
                variables.append(var_info)
        
        return variables
    
    def parse_single_variable(self, type_str, var_str, line_num):
        type_str = type_str.strip()
        var_str = var_str.strip()
        
        initialized = '=' in var_str
        
        if '=' in var_str:
            var_name_part, init_part = var_str.split('=', 1)
            var_name_part = var_name_part.strip()
            init_value = init_part.strip()
        else:
            var_name_part = var_str
            init_value = None
        
        array_suffix = ""
        if '[' in var_name_part:
            array_match = re.search(r'(\[[^\]]*\])', var_name_part)
            if array_match:
                array_suffix = array_match.group(1)
                var_name_part = var_name_part[:var_name_part.find('[')]
        
        var_name_part = var_name_part.strip()
        pointer_count = var_name_part.count('*')
        var_name = var_name_part.replace('*', '').strip()
        
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var_name):
            return None
        
        full_type = type_str
        if pointer_count > 0:
            full_type += '*' * pointer_count
        full_type += array_suffix
        
        if not full_type.strip():
            full_type = "int"
        
        return {
            'name': var_name,
            'type': full_type.strip(),
            'initialized': initialized,
            'init_value': init_value,
            'line_number': line_num
        }
    
    def find_global_variable_usage(self):
        for data_ref in self.global_vars:
            for func in self.functions:
                pattern = r'(?<!\w)' + re.escape(data_ref.name) + r'(?!\w)'
                if re.search(pattern, func['full_function']):
                    data_ref.parents.append(func['name'])

    def get_weight_children(self, function_name):
        if function_name not in self.call_graph:
            return 0
        visited = set()
        count = 0
        
        def get_children(func_name):
            nonlocal count
            if func_name in visited:
                return
            visited.add(func_name)
            
            for child_func in self.call_graph[func_name]:
                count += 1
                get_children(child_func)
        
        get_children(function_name)
        return count

    def get_weight_parents(self, function_name):
        if function_name not in self.call_graph:
            return 0
        
        visited = set()
        count = 0
        
        def get_parents(func_name):
            nonlocal count
            if func_name in visited:
                return
            visited.add(func_name)

            for parent_func in self.call_graph:
                if func_name in self.call_graph[parent_func]:
                    count += 1
                    get_parents(parent_func)
        
        get_parents(function_name)
        return count
        

def main():
    if len(sys.argv) != 2:
        print("Usage: python test.py <c_file>")
        return
    filename = sys.argv[1]
    parser = CFunctionParser(filename)
    print("call graph")
    for i in parser.call_graph:
        print(i)
        print("->",parser.call_graph[i])
    print("-"*100)
    print("functions")
    for i in parser.functions:
        print(i)
    print("-"*100)
    print("global variables")
    # for var in parser.global_vars:
        # print(var)
        # print(f"Name: {var.name}")
        # print(f"Type: {var.type}")
        # print(f"Initialized: {var.initialized}")
        # print(f"Init Value: {var.init_value}")
        # print(f"Line Number: {var.line_number}")
        # print(f"Used by functions: {var.parents}")
        # print("-" * 50)
if __name__ == "__main__":
    main()