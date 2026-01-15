import sys
import argparse
from elftools.elf.elffile import ELFFile

def dump_data_from_vaddr(
    binary_path: str,
    virtual_address: int,
    length: int,
    section_name: str
) -> bytes:
    """
    Calculates a file offset from a virtual address and dumps data of a given length.

    Args:
        binary_path: Path to the ELF binary file.
        virtual_address: The virtual memory address to read from.
        length: The number of bytes to dump.
        section_name: The name of the section containing the address (e.g., '.rodata').

    Returns:
        The raw bytes read from the file.

    Raises:
        ValueError: If the section isn't found or the read range is out of bounds.
        FileNotFoundError: If the binary_path does not exist.
    """
    try:
        with open(binary_path, 'rb') as f:
            elf = ELFFile(f)
            section = elf.get_section_by_name(section_name)

            if section is None:
                raise ValueError(f"Section '{section_name}' not found.")

            sec_addr = section['sh_addr']
            sec_offset = section['sh_offset']
            sec_size = section['sh_size']

            # Check if the requested data range is valid within the section
            if not (sec_addr <= virtual_address < sec_addr + sec_size):
                raise ValueError(f"Start address 0x{virtual_address:x} is not in section '{section_name}'.")
            if (virtual_address + length) > (sec_addr + sec_size):
                raise ValueError(f"Read of length {length} would go past the end of section '{section_name}'.")

            # Calculate the final file offset
            file_offset = (virtual_address - sec_addr) + sec_offset

            # Seek to the offset and read the data
            f.seek(file_offset)
            data = f.read(length)
            return data

    except FileNotFoundError:
        print(f"Error: Binary file not found at '{binary_path}'", file=sys.stderr)
        raise
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise

def format_hexdump(data: bytes, start_addr: int = 0) -> str:
    """Formats raw bytes into a readable hexdump string."""
    lines = []
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        # Format the hexadecimal part, ensuring it has a fixed width
        hex_part = ' '.join(f'{b:02x}' for b in chunk)
        # Format the ASCII part, replacing non-printable characters
        ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        lines.append(f"{start_addr + i:08x}: {hex_part:<48} |{ascii_part}|")
    return "\n".join(lines)

def main():
    """Main function to parse arguments and execute the dump."""
    parser = argparse.ArgumentParser(
        description="Dump data from a virtual address in an ELF file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "binary_path",
        help="Path to the ELF binary file."
    )
    parser.add_argument(
        "virtual_address",
        # Allow hex (0x...), octal (0o...), or decimal addresses
        type=lambda x: int(x, 0),
        help="The virtual memory address to read from (e.g., 0x401000)."
    )
    parser.add_argument(
        "length",
        type=int,
        help="The number of bytes to dump."
    )
    parser.add_argument(
        "section_name",
        help="The name of the section containing the address (e.g., '.rodata')."
    )

    args = parser.parse_args()

    try:
        # Dump the raw data from the specified address
        raw_data = dump_data_from_vaddr(
            args.binary_path,
            args.virtual_address,
            args.length,
            args.section_name
        )
        
        # Format the data into a hexdump, starting from the requested virtual address
        hexdump_output = format_hexdump(raw_data, start_addr=args.virtual_address)
        
        # Print the final result to standard output
        print(hexdump_output)
        
    except (ValueError, FileNotFoundError):
        # Errors are already printed to stderr by the function, so just exit
        sys.exit(1)

if __name__ == "__main__":
    main()
