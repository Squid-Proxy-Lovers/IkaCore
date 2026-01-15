import re
from pathlib import Path

DUMP = """
00000000: 6944 6137 0148 6973 746f 7279 3634 0030  iDa7.History64.0
00000010: 001d 0000 0001 2f68 6f6d 652f 6374 6670  ....../home/ctfp
00000020: 6c61 7965 722f 6374 665f 6669 6c65 732f  layer/ctf_files/
00000030: 6170 7031 0042 0000 0001 2f68 6f6d 652f  app1.B..../home/
00000040: 7370 6c2f 6167 656e 7469 632d 6c6c 6d2d  spl/agentic-llm-
00000050: 6373 6177 2d63 6f6d 702f 4354 4654 696e  csaw-comp/CTFTin
00000060: 792f 6374 6674 696e 792f 7265 762f 6261  y/ctftiny/rev/ba
00000070: 6279 5f6d 756c 742f 6368 616c 0001 4c69  by_mult/chal..Li
00000080: 6365 6e73 6573 0001 4944 4150 524f 2e69  censes..IDAPRO.i
00000090: 6461 2d70 726f 0030 004c 0000 0001 7b22  da-pro.0.L....{"
000000a0: 6c69 6373 7263 223a 7b22 7061 7468 223a  licsrc":{"path":
000000b0: 222f 6f70 742f 6964 612d 7072 6f2d 392e  "/opt/ida-pro-9.
000000c0: 312f 6964 6170 726f 2e68 6578 6c69 6322  1/idapro.hexlic"
000000d0: 7d2c 226c 6964 223a 2234 382d 3231 3337  },"lid":"48-2137
000000e0: 2d41 4341 422d 3939 227d 0000 4555 4c41  -ACAB-99"}..EULA
000000f0: 2039 3000 0400 0000 0401 0000 004c 6963   90..........Lic
00000100: 656e 7365 5072 6576 5761 726e 5469 6d65  ensePrevWarnTime
00000110: 0004 0000 0004 004c ca68 5079 7468 6f6e  .......L.hPython
00000120: 3354 6172 6765 7444 4c4c 002e 0000 0001  3TargetDLL......
00000130: 2f75 7372 2f6c 6962 2f78 3836 5f36 342d  /usr/lib/x86_64-
00000140: 6c69 6e75 782d 676e 752f 6c6962707974  linux-gnu/libpyt
00000150: 686f 6e33 2e31 302e 736f 2e31 2e30 5365  hon3.10.so.1.0Se
00000160: 6172 6368 4269 6e00 0000 0000 0153 6561  archBin......Sea
00000170: 7263 6854 6578 7400 0000 0000 0175 6943  rchText......uiC
00000180: 6f6e 6669 6736 3400 6800 0000 03ff ffff  onfig64.h.......
00000190: ffff ffff ff00 0000 0000 0000 0000 0000  ................
000001a0: 0000 0000 0002 0000 0001 0000 0000 0000  ................
000001b0: 00ff ffff ff00 0001 00ff ffff ffff ffff  ................
000001c0: ff00 0000 0000 0000 0000 0000 0000 0000  ................
000001d0: 0002 0000 0000 0000 0000 0000 0000 0000  ................
000001e0: 0000 0000 004d 0001 0500 0000 0000 0000  .....M..........
000001f0: 0000 0000 00bf 308b dd                   ......0..
"""

def parse_xxd_dump(dump: str) -> bytes:
    out = bytearray()
    for line in dump.splitlines():
        line = line.rstrip()
        if not line or ':' not in line:
            continue
        # Split off the offset
        _, rest = line.split(':', 1)
        # Keep only the hex column (before the ASCII area which starts after two spaces)
        hex_col = rest.split('  ')[0]
        # Extract byte pairs
        pairs = re.findall(r'[0-9A-Fa-f]{2}', hex_col)
        if pairs:
            out.extend(bytes.fromhex(''.join(pairs)))
    return bytes(out)

if __name__ == "__main__":
    data = parse_xxd_dump(DUMP)
    out_path = Path("ida.reg")
    out_path.write_bytes(data)
    print(f"[+] Wrote {len(data)} bytes to {out_path.resolve()}")
