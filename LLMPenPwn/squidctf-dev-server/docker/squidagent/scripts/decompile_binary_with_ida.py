import os
import rpyc
from headless_ida import HeadlessIda

IDA_PATH = "/opt/ida-pro-9.1/idat"
import sys

if len(sys.argv) < 2:
    print("Usage: python decompile_binary_with_ida.py <binary_path> [output_file]")
    sys.exit(1)

binary_path = sys.argv[1]
if len(sys.argv) >= 3 and sys.argv[2].strip():
    output_file = sys.argv[2]
else:
    output_file = f"{os.path.splitext(os.path.basename(binary_path))[0]}_decompiled.c"

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

print(f"[+] Launching IDA at {IDA_PATH} on {binary_path}")

try:
    ida = HeadlessIda(IDA_PATH, binary_path)
    _exit_fn = None
    # Select best available lifecycle
    if hasattr(ida, "__enter__") and hasattr(ida, "__exit__"):
        ida.__enter__()
        def _exit():
            ida.__exit__(None, None, None)
        _exit_fn = _exit
    elif hasattr(ida, "start"):
        ida.start()
        _exit_fn = getattr(ida, "stop", None)
    elif hasattr(ida, "open"):
        ida.open()
        _exit_fn = getattr(ida, "close", None)
    elif hasattr(ida, "launch"):
        ida.launch()
        _exit_fn = getattr(ida, "terminate", None)
    try:
        import idaapi
        import ida_hexrays
        import ida_funcs
        import idautils
        import idc
        import ida_bytes
        import ida_name

        def ensure_hexrays():
            if not ida_hexrays.init_hexrays_plugin():
                print("[!] Hex-Rays decompiler is not available.")
                return False
            return True

        def decompile_function(ea):
            try:
                cfunc = ida_hexrays.decompile(ea)
                if cfunc:
                    return str(cfunc)
            except ida_hexrays.DecompilationFailure:
                return None
            return None

        def is_useful_reference(addr, name, data_content=None):
            if addr < 0x1000 and (not name or name.startswith("DAT_")):
                return False
            if name and any(elf_term in name.lower() for elf_term in ["elf", "dword_0", "dat_0", "dat_1", "dat_2", "dat_3", "dat_4", "dat_5", "dat_6", "dat_7"]):
                return False
            if name and any(ptr_term in name.lower() for ptr_term in ["ptr", "qword", "dword"]):
                return False
            size = ida_bytes.get_item_size(addr)
            if size <= 2 and (not name or name.startswith("DAT_")):
                return False
            if size in [4, 8]:
                data = ida_bytes.get_bytes(addr, size)
                if data:
                    value = int.from_bytes(data, "little")
                    if value > 0x400000:
                        return False
            if data_content and isinstance(data_content, str):
                if len(data_content.strip()) <= 2:
                    return False
            return True

        def get_data_declaration(addr):
            name = idc.get_name(addr, ida_name.GN_VISIBLE)
            if not name:
                name = f"DAT_{addr:X}"
            s = idc.get_strlit_contents(addr, -1, idc.STRTYPE_C)
            if s:
                try:
                    s = s.decode("utf-8", errors="replace")
                except AttributeError:
                    s = str(s)
                if not is_useful_reference(addr, name, s):
                    return None
                s = s.replace('"', '\\"')
                return f'const char {name}[] = "{s}";\n'
            if not is_useful_reference(addr, name):
                return None
            size = ida_bytes.get_item_size(addr)
            if size <= 0:
                size = 8
            data = ida_bytes.get_bytes(addr, size)
            if not data:
                return None
            hex_bytes = ", ".join(f"0x{b:02X}" for b in data)
            return f"unsigned char {name}[{size}] = {{{hex_bytes}}}; // at 0x{addr:X}\n"

        def collect_all_useful_data_references():
            references = {}
            for ea in idautils.Functions():
                for head in idautils.FuncItems(ea):
                    if idc.is_code(idc.get_full_flags(head)):
                        for op in range(idaapi.UA_MAXOP):
                            opnd = idc.get_operand_value(head, op)
                            if opnd != idc.BADADDR:
                                seg = idaapi.getseg(opnd)
                                if seg and seg.type != idaapi.SEG_CODE:
                                    if opnd not in references:
                                        decl = get_data_declaration(opnd)
                                        if decl:
                                            references[opnd] = decl
            return references

        print("[+] Testing connection...")
        function_count = 0
        sample_functions = []
        for func in idautils.Functions():
            function_count += 1
            if len(sample_functions) < 5:
                sample_functions.append(f"{hex(func)} {ida_name.get_ea_name(func)}")
        print(f"[+] Connection successful! Found {function_count} functions")
        for func_info in sample_functions:
            print(f"    {func_info}")

        print("[+] Checking Hex-Rays decompiler availability...")
        if not ensure_hexrays():
            raise RuntimeError("Hex-Rays decompiler is not available")

        print("[+] Starting decompilation process...")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("// Decompiled output generated by HeadlessIda + Hex-Rays\n")
            f.write(f"// Source binary: {binary_path}\n")
            f.write("// Filtered global data references\n\n")
            print("[+] Collecting global data references...")
            all_data = collect_all_useful_data_references()
            if all_data:
                f.write("// ===== Global Data References =====\n\n")
                for addr, decl in sorted(all_data.items()):
                    f.write(decl)
                f.write("\n\n")
                print(f"[+] Found {len(all_data)} useful data references")
            else:
                print("[+] No useful data references found")

            functions = list(idautils.Functions())
            print(f"[+] Processing {len(functions)} functions...")
            successful_decompilations = 0
            for i, ea in enumerate(functions, 1):
                func_name = idc.get_func_name(ea)
                if i % 10 == 0 or i == len(functions):
                    print(f"[+] Processing {i}/{len(functions)}: {func_name}")
                f.write(f"// Function: {func_name} (0x{ea:X})\n")
                code = decompile_function(ea)
                if code:
                    f.write(code)
                    f.write("\n\n" + ("-" * 80) + "\n\n")
                    successful_decompilations += 1
                else:
                    f.write(f"// Failed to decompile {func_name}\n\n")
                    f.write("-" * 80 + "\n\n")

        print("[+] Decompilation complete!")
        print(f"[+] Output saved to: {output_file}")
    finally:
        if _exit_fn:
            _exit_fn()
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[!] Error during decompilation: {e}")

