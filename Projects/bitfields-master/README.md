# bitfields
Packing and extracting bit fields in Python integers. 
This is the first in a series of projects in which we will build and program a simulated computer, the Duck Machine 2018S, or DM2018S.  

## Context: Why we need bitfields

Executable *machine code* instructions for a computer are stored in memory in binary, in a series of memory *words*.  Words in the DM2018S consist of 32 binary digits (bits), or 4 bytes.   Memory is uniform and untyped:  The same memory word could be interpreted as an integer or as an instruction or a few characters of text or part of a floating point number.  Treating it as an instruction is simply a matter of how we use it. 

Each instruction for the computer needs to specify several things: An operation to be performed, operand values or the locations of operand values, where to put the result of the operation, and sometimes some additional information. It would be inefficient to keep each part of that information in a separate memory word.  Instead, all modern computers *pack* this information into *fields* in a smaller number of words.  Many computers following the *RISC* design style pack all the fields of an instruction into a single memory word.  The processor in your phone is likely based on RISC design principles. The DM2018S is also a RISC machine, based loosely on the popular ARM processor used in many cell phones and in the Raspberry Pi. 

Each of the fields of an instruction (e.g., the operation code, and the index of the register that should receive the result of the operation) can be represented in a few binary digits.  For example, if there are fewer than 32 (2^5) operation codes, then they can be represented by a 5-bit binary number.  If the number of bits required to represent each field sum to 32 or less, then we can *pack* all of them into a 32-bit memory word.  That is what we will do with DM2018S instructions. 

## BitField objects

A BitField object is a tool for inserting a field into an integer or extracting a field from an integer. 

For example, suppose we wanted to keep some values in the first four bits (bits 0..3) and other values in the second four bits (bits 4..7) of integers.  We would define two BitField objects: 

```
low4 = BitField(0,3)
mid4 = BitField(4,7)
```

Then we could pack two small numbers, each between 0 and 15 inclusive, into one integer: 

```
packed = 0
packed = low4.insert(7, packed)
packed = mid4.insert(8, packed)
```

At this point the value of packed is 135, but let's not think about it in decimal.  Think in hex, where each hexadecimal digit represents 4 binary bits.  Now the value is clear: 

```
>>> hex(packed)
'0x87'
```
There is our 7, in the low digit, and our 8, in the high digit.  

We can also use the BitField object to extract the individual fields: 

```
>>> low4.extract(packed)
7
>>> mid4.extract(packed)
8
```

## Putting BitFields to Work

The arithmetic logic unit (ALU) is the part of the central processing unit (CPU) that actually performs arithmetic operations.  There is more to executing a CPU instruction than performing that operations (e.g., we need a way of sequencing operations, and we need a way of moving data between the CPU and main memory); we will build those parts in the next project.  For now, the ALU is enough to put BitField objects to work. 

We can see how DM2018S instructions are packed into memory words in instr_format.py: 

```
reserved = BitField(31,31)
instr_field = BitField(26, 30)
cond_field = BitField(22, 25)
reg_target_field = BitField(18, 21)
reg_src1_field = BitField(14, 17)
reg_src2_field = BitField(10, 13)
offset_field = BitField(0, 9)
```

```instr_field``` will hold an instruction code, which is defined by a Python enumeration class:

```
class OpCode(Enum):
    """The operation codes specify what the CPU and ALU should do."""
    # CPU control (beyond ALU)
    HALT = 0    # Stop the computer simulation (in Duck Machine project)
    LOAD = 1    # Transfer from memory to register
    STORE = 2   # Transfer from register to memory
    # ALU operations
    ADD = 3     # Addition
    SUB = 5     # Subtraction
    MUL = 6     # Multiplication
    DIV = 7     # Integer division (like // in Python)
```
Since there are only 8 operation codes, we could have reserved as few as 3 binary digits to hold an operation code, but we have allocated 5 bits so that up to 32 operation codes could be accommodated in the future. 

The DM2018S has 16 *registers*, which are like very fast memory cells built into the CPU itself (in contrast to main memory, which is a separate hardware component).   The ALU does not access registers directly, but the values fed into the ALU come from registers, and the result value produced by the ALU is then stored in a register.  Since there are 16 registers, we need 4 binary digits to specify each of the two source registers and the target register: 

```
reg_target_field = BitField(18, 21)
reg_src1_field = BitField(14, 17)
reg_src2_field = BitField(10, 13)
```

In addition, the instruction holds a 10 bit integer called the *offset*, which is added to the second source register before it is fed to the ALU.  While the other fields are interpreted as *unsigned* integers, the offset field is a *signed* integer, i.e., it may be positive or negative.  Thus its highest digit, bit 9, is 1 if the number is negative and 0 if the number is positive.  When we extract this number from an instruction, it will be necessary to *sign extend* negative values to create valid negative Python integers. 

One bit of the DM2018S instruction word is reserved for future expansion.  There is also a *condition* field, which will be important in the next project for controlling loops in DM2018S machine code programs, but we will ignore it for now. 

Since our ALU is not yet part of a complete CPU, we have a very small driver program that executes a very short DM2018S program.  Here's the program: 

```
264503311
264782848
399278085
466403330
```
Really, that's the program.  Each of those 32-bit integers contains the fields of a DM2018S instruction, packed into bit fields.  We need to unpack or *decode* them, one at a time, to execute the program.  In lieu of a whole CPU to do that for us, our program ```example_tiny_run.py``` drives the process: 

```
    for word in words:
        instr = instr_format.decode(word)
        op = instr.op
        to_reg = instr.reg_target
        src_1 = instr.reg_src1
        src_2 = instr.reg_src2
        offset = instr.offset
        result, condition = alu.exec(op, 
            regs[src_1], regs[src_2] + offset)
        regs[to_reg] = result
```

```instr_format.decode``` uses bitfield objects to extract each of the instruction fields from the word and form an Instruction object, which guides how we call ```alu.exec```.      

But what does this program mean?  Here is how it is created by another tiny program, ```example_tiny_create.py```, from a sequence of Instruction objects: 

```
def gen(s: str) -> int:
    """Create the integer encoding of one instruction from a string"""
    instr = instruction_from_string(s)
    encoding = instr.encode()
    return encoding

def make_sample() -> List[int]:
    """Create a small, arbitrary little sequence of
    instructions encoded as 32-bit binary integers.
    """
    return  [
        gen("ADD  ALWAYS r1  r0 r0  15"), # Initializes r1 to 15
        gen("ADD  ALWAYS r2  r1 r1  0"),  # r2 = r1 + r1
        gen("SUB  ALWAYS r3  r2 r0  5"),  # r3 = r2 - 5
        gen("MUL  ALWAYS r3  r3 r0  2")   # r3 = r3 * 2for
    ]
```

We expect ```example_tiny_run.py``` to essentially reverse the encoding process, recovering and executing these instructions.  If it works correctly, we will see the following log output from ```example_tiny_run.py```:

```
INFO:__main__:Executing ADD      r1,r0,r0[15]
INFO:__main__:Computed value 15
INFO:__main__:Executing ADD      r2,r1,r1[0]
INFO:__main__:Computed value 30
INFO:__main__:Executing SUB      r3,r2,r0[5]
INFO:__main__:Computed value 25
INFO:__main__:Executing MUL      r3,r3,r0[2]
INFO:__main__:Computed value 50
```

It is not a very exciting program, just computing some values in registers, but it's about the best we can do with just an arithmetic logic unit. 

## Project Steps

This is a project that requires more thinking than coding.  Be sure you really understand how each part works.  If some parts don't make sense, ask us about them! 

Start by completing BitFields class.  There is a test program, test_bitfields.py, to help you determine whether your BitFields class is working.  The test program may also help you clarify questions about exactly how BitFields objects are used and their expected behavior. 

Completing BitFields project will require very little code.  However, it is code that can require very careful thinking, and needs to be well-documented.  For example, my implementation of method BitField.insert has only four lines of code, three lines of comments (excluding the docstring), and a six-line docstring including an example.  That's a high ratio of documentation to code! 

When the BitFields class is working, then you can move on to finishing the ALU class.  In hardware, the ALU would use [multiplexer](https://en.wikipedia.org/wiki/Multiplexer) and demultiplexer circuits to direct its operands through a circult determined by the operation code.  A software analogue is to look up the function to apply in a table of functions.  This is not very complex, and it isn't much code, but it may be something you are not very familiar with, so I have provided part of the table and left part of it for you to fill in. 

There is a test program for the ALU.  When you have completed the ALU and can pass those tests, you should also be able to run ```tiny_example_run.py```. 

