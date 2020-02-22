"""
Duck Machine model DM2018S CPU
"""

from instr_format import Instruction, OpCode, CondFlag, decode
from register import Register, ZeroRegister
from alu import ALU
from mvc import MVCEvent, MVCListenable

import logging

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class CPUStep(MVCEvent):
    """CPU is beginning step with PC at a given address"""

    def __init__(self, subject: "CPU", pc_addr: int,
                 instr_word: int, instr: Instruction) -> None:
        self.subject = subject
        self.pc_addr = pc_addr
        self.instr_word = instr_word
        self.instr = instr


class CPU(MVCListenable):
    """Duck Machine central processing unit (CPU)
    has 16 registers (including r0 that always holds zero
    and r15 that holds the program counter), a few
    flag registers (condition codes, halted state),
    and some logic for sequencing execution.  The CPU
    does not contain the main memory but has a bus connecting
    it to a separate memory.
    """

    def __init__(self, memory):
        super().__init__()
        self.memory = memory  # Not part of CPU; what we really have is a connection
        self.registers = [ ZeroRegister(), Register(), Register(), Register(),
                           Register(), Register(), Register(), Register(),
                           Register(), Register(), Register(), Register(),
                           Register(), Register(), Register(), Register() ]
        self.condition = CondFlag.ALWAYS
        self.halted = False
        self.alu = ALU()
        # Convenient aliases
        self.pc = self.registers[15]

    def step(self):
        log.debug("Step at PC={}".format(self.pc.get()))

        # Fetch
        instr_addr = self.pc.get()
        instr_word = self.memory.get(instr_addr)

        # Decode
        instr = decode(instr_word)
        log.debug("Instruction: {}".format(instr))
        # Display the CPU state when we have decoded the instruction,
        # before we have executed it
        self.notify_all(CPUStep(self, instr_addr, instr_word, instr))

        # Execute
        predicate = instr.cond
        if predicate & self.condition:
            log.debug("Predicate passed")
            opcode = instr.op
            # Add code below: Set target, left, and right appropriately
            target = self.registers[instr.reg_target]
            left = self.registers[instr.reg_src1].get()
            right = self.registers[instr.reg_src2].get() + instr.offset
            # Step program counter after forming operands but before
            # storing execution result
            self.pc.put(self.pc.get() + 1)
            # Now a store into PC will overwrite the stepped value
            result, cc = self.alu.exec(opcode, left, right)
            self.condition = cc
            # Load and store are special
            if opcode == OpCode.LOAD:
                memval = self.memory.get(result)
                target.put(memval)
            elif opcode == OpCode.STORE:
                self.memory.put(result,target.get())
            elif opcode == OpCode.HALT:
                self.halted = True
            else:
                target.put(result)
        else:
            # The program counter still moves forward, with no
            # other computation
            log.debug("Predicated instruction will not execute")
            self.pc.put(self.pc.get() + 1)

    def run(self, from_addr = 0, single_step = False) -> None:
        """A loop that calls the step method repeatedly"""

        self.halted = False
        self.pc.put(from_addr)
        step_count = 0

        while not self.halted:
            self.step()
            step_count += 1

            if single_step:
                input("Step {}; press enter".format(step_count))


# Create a class CPU, subclassing MVCListenable.
# It should have 16 registers (a list of Register objects),
# and the first of them should be the special ZeroRegister
# object that is always zero regardless of what is stored.
# It should have a CondFlag with the current condition.
# It should have a boolean "Halted" flag, and execution of
# the "run" method should halt with the Halted flag is True
# (set by the HALT instruction). The CPU does not contain
# the memory, but has a connection to a Memory object
# (specifically a MemoryMappedIO object).
# See the project web page for more guidance.

