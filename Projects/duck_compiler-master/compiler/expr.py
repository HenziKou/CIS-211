"""
Expressions.  An expression is a subtree, which may be
- a numeric value, like 5
- a named variable, like x
- a binary operator, like 'plus', with a left and right subtree
- an assignment:  evaluate right hand side, store in variable in left hand side
- a control flow operator, like 
   - pass: do nothing
   - sequence: do one thing, and then another
   - if/then/else:  test a condition and then execute one branch or another
   - while: test a condition to control a loop
   Where there is a condition, we treat 0 as False and any other value
   as true. 

In addition to the new control flow operators, the calculator is extended
for Duck Machine assembly code generation.  The 'eval' methods evaluate an 
expression immediately, while the 'gen' methods create assembly language 
that can be translated into object code to evaluate the expression.

Author: Henzi Kou

"""

# Python standard libraries
from numbers import Real

# Our modules
from compiler.env import Env
from compiler.codegen_context import Context

import logging

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Expr(object):
    """Abstract base class. Cannot be instantiated."""

    def eval(self, env: Env) -> "Const":
        """Each concrete subclass of Expr must define this method"""
        raise NotImplementedError(
            "No eval method has been defined for class {}".format(type(self)))

    def gen(self, context: Context, target: str):
        """Code generation.  Walk the expression and build
        the instruction stream in the context object.  If a 
        target is given, it is the name of the register into 
        which the generated code should place a value. 
        """
        raise NotImplementedError(
            "No gen method has been defined for class {}".format(type(self)))


class Const(Expr):
    """An expression that is just a constant value, like 5"""

    def __init__(self, value: int):
        self.val = value

    def value(self):
        """The internal value"""
        return self.val

    def __repr__(self):
        return "Const({})".format(self.val)

    def __str__(self):
        return str(self.val)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.val == other.val

    def eval(self, env: Env) -> "Const":
        """This is about as evaluated as it can get"""
        log.debug("Evaluating {} in Const".format(self))
        return self

    def gen(self, context: Context, target: str):
        """Load a constant from memory into a register"""
        const_label = context.get_const_symbol(self.val)
        context.add_line("\tLOAD {},{}  # Const {}".format(target, const_label, self.val))


# It's handy to have a special singleton value for things that are undefined, and another
# for things that default to zero
NO_VALUE = Const(-97979797)
ZERO = Const(0)


class Var(Expr):
    """A variable has a name and may have a value in the environment."""

    def __init__(self, name):
        """Expression is reference to a variable named name"""
        assert isinstance(name, str)
        self.name = name

    def eval(self, env: Env) -> Const:
        """Fetches value from environment."""
        log.debug("Evaluating {} in Var".format(self))
        val = env.get(self.name)
        log.debug("Returning {}".format(val))
        return val

    def __repr__(self):
        return "Var('{}')".format(self.name)

    def __str__(self):
        return self.name

    def gen(self, context: Context, target: str):
        """Code generation for a variable reference.
        Generates code to load the value of that variable
        from memory.
        """
        log.debug("Generating code for reference to variable {}"
                  .format(self.name))
        symbol = context.get_var_symbol(self.name)
        context.add_line("\tLOAD {},{}".format(target, symbol))
        return


# noinspection PyAbstractClass
class Control(Expr):
    """Control flow nodes (while, if, ...).
    Control flow constructs have one or more blocks of statements
    and may have a controlling predicate.  For predicates,
    we take zero as false, and any other value as true.
    Control constructs don't have actual values (they would be 'None'
    in Python and 'void' in C or C++), so we return NO_VALUE
    from eval.
    """
    pass
    # Note PyCharm will complain that Control doesn't implement all
    # abstract methods, but that's because Control is itself an
    # abstract base class ... the abstract methods should be implemented
    # in its subclasses.


class Seq(Control):
    """exp ; exp"""

    def __init__(self, left, right):
        """ exp ; exp """
        self.left = left
        self.right = right

    def __str__(self):
        return "{{\n{}\n{} }}".format(self.left, self.right)

    def eval(self, env: Env) -> Const:
        """Just evaluate in order"""
        discard = self.left.eval(env)
        discard = self.right.eval(env)
        return NO_VALUE

    def gen(self, context: Context, target: str):
        """Just execute the statements in order.
        Discard the results, if any.
        """
        log.debug("Generating code for sequence")
        self.left.gen(context, target)
        self.right.gen(context, target)


class While(Control):
    """Classic while loop; in postfix we will write
    while cond do exp as exp cond while
    """

    def __init__(self, cond, expr):
        """While cond do expr"""
        self.cond = cond
        self.expr = expr

    def __str__(self):
        return "while {} do\n{}\nod".format(self.cond, self.expr)

    def eval(self, env: Env) -> Const:
        """
        Repeat 'expr' part while 'cond' part evaluates to a non-zero
        value.  Always returns 0 (which will be discarded)
        """
        cond_val = self.cond.eval(env)
        while cond_val.value() != 0:
            discard = self.expr.eval(env)
            cond_val = self.cond.eval(env)
        return Const(0)

    def gen(self, context: Context, target: str):
        """Translate 'while' loop into explicit jumps.
        """
        loop_head = context.new_label("loop")
        loop_exit = context.new_label("endloop")
        context.add_line("{}:  #While loop".format(loop_head))
        reg = context.alloc_reg()
        self.cond.gen(context, target=reg)
        # Is it zero?
        context.add_line("\tSUB  r0,{},r0 ".format(reg))
        context.add_line("\tJUMP/Z {}".format(loop_exit))
        context.free_reg(reg)
        self.expr.gen(context, target)
        context.add_line("\tJUMP {}".format(loop_head))
        context.add_line("{}: ".format(loop_exit))


class Pass(Control):
    """
    The 'else' part of an 'if' statement is optional.  This node
    is a stand-in for the empty block ... it does nothing.
    """

    def __init__(self):
        """La la la la la I can't hear you"""
        return

    def __repr__(self):
        return "pass"

    def __str__(self):
        return "pass"

    def eval(self, env: Env) -> Const:
        """Does nothing, has no value."""
        return NO_VALUE

    def gen(self, context: Context, target: str):
        """
        Pass does nothing, has no value.
        """
        return


class If(Control):
    """If with optional Else (no elif)"""

    def __init__(self, cond, thenpart, elsepart=Pass()):
        """While cond do block"""
        self.cond = cond
        self.thenpart = thenpart
        self.elsepart = elsepart

    def __str__(self):
        return "if {} then\n{}\nelse\n{}\nfi".format(self.cond, self.thenpart, self.elsepart)

    def eval(self, env: Env) -> Const:
        """If statement.  Returns nothing. """
        cond_value = self.cond.eval(env)
        if cond_value.value() != 0:
            discard = self.thenpart.eval(env)
        else:
            discard = self.elsepart.eval(env)
        return NO_VALUE

    def gen(self, context: Context, target: str):
        """
        Generate code for an if/else.
        """

        # The outline of the code you should generate is:
        # <code for expression>
        # subtract expression result from zero
        # if zero, jump to elsepart
        # <code for 'then' part>
        # jump to endif
        # elsepart:
        # <code for elsepart>
        # fi:
        # Generate fresh labels for the 'elsepart' and 'fi' each time,
        # since there could be more than one 'if' statement in a program.
        # Look at the 'while' statement above for examples of code
        # generation for tests, jumps, and labels.
        #
        # Example:
        # if 10 - 9:    --->   condition
        #    ------
        #   x = 5       --->   then
        # else:
        #   x = 0       --->   else

        # initialize labels and registers
        elsepart = context.new_label("else")
        fi = context.new_label("end else")
        reg = context.alloc_reg()
        self.cond.gen(context, target=reg)          # generate condition

        # If part
        # Is it zero?
        context.add_line("\tSUB  r0,{},r0 ".format(reg))
        context.add_line("\tJUMP/Z {}".format(elsepart))
        self.thenpart.gen(context, target)          # generate then part
        context.add_line("JUMP {}".format(fi))

        # Else part
        context.add_line("{}:  #Else loop".format(elsepart))
        self.elsepart.gen(context, target)          # generate else part
        context.add_line("{}".format(fi))
        context.free_reg(reg)                       # free the register


class Assign(Expr):
    """x = Expr.  Evaluated for side-effect;
    returns NO_VALUE.
    """

    def __init__(self, var, expr):
        """Representation of 'let var = expr'"""
        assert isinstance(var, Var)
        assert isinstance(expr, Expr)
        self.var = var
        self.expr = expr

    def __repr__(self):
        return "Assign({},{})".format(self.var, self.expr)

    def __str__(self):
        return "let {} = {}".format(self.var, self.expr)

    def eval(self, env: Env) -> Const:
        """Stores value of expr (evaluated) in environment"""
        log.debug("Evaluating {} in Assign".format(self))
        val = self.expr.eval(env)
        env.put(self.var.name, val)
        return NO_VALUE

    def gen(self, context: Context, target: str):
        """Code generation for assignment: calculate into register,
        then store into memory
        """
        log.debug("Generating code for assignment")
        var_symbol = context.get_var_symbol(self.var.name)
        self.expr.gen(context, target)
        context.add_line("\tSTORE  {},{}".format(target, var_symbol))


class BinOp(Expr):
    """Abstract superclass for binary expressions like plus, minus"""

    def __init__(self, left, right):
        """A binary operation has a left and right sub-expression"""
        assert isinstance(left, Expr)
        assert isinstance(right, Expr)
        self.left = left
        self.right = right

    def __eq__(self, other):
        """Identical expression"""
        return type(self) == type(other) \
            and self.left == other.left \
            and self.right == other.right

    def eval(self, env: Env) -> Const:
        """Evaluation strategy for binary operations
        that apply to numbers and produce numbers.
        """
        log.debug("Evaluating {} in BinOp".format(self))
        lval = self.left.eval(env)
        assert isinstance(lval, Const), "Op {} applies to numbers, not to {}".format(
            type(self).__name__, lval)
        lval_n = lval.value()
        rval = self.right.eval(env)
        assert isinstance(lval, Const), "Op {} applies to numbers, not to {}".format(
            type(self).__name__, rval)
        rval_n = rval.value()
        return Const(self._apply(lval_n, rval_n))

    def _apply(self, left: int, right: int) -> int:
        """Apply operation to numeric values.  Each concrete
        subclass of BinOp must define this method.
        Note: In Python, 'int' and 'float' are subtypes of Real.
        """
        raise NotImplementedError(
            "Class {} has not defined its _apply method".format(type(self)))

    def gen(self, context: Context, target: str):
        """Code generation for an binary operations
        """
        log.debug("Generating code for binary operator")

        #    You can use the same register for the left operand as
        #    the 'target' register passed in.  For the right operand,
        #    you need to allocate a new register to pass as the
        #    'target' of the right operand.
        #    After you have evaluated left and right operands,
        #    generate an instruction that looks like
        #        OP  target,target,other_register
        #    where OP is the operation code (like ADD, SUB, etc)
        #    for the particular binop.  Subclasses Plus, Minus,
        #    etc do not repeat this 'gen' method, but instead just
        #    define the _opcode method to provide the needed
        #    operation code.
        #    After generating code for this operation, be sure to
        #    free the register you allocated for the right operand.

        self.left.gen(context, target)
        reg = context.alloc_reg()
        self.right.gen(context, target=reg)

        context.add_line("\t{} {},{},{}".format(self._opcode(), target, target, reg))
        context.free_reg(reg)                       # free the register


    def _opcode(self):
        """Each operation that inherits gen must provide the opcode
        to be used in the instruction. 
        """
        raise NotImplementedError("Class {} doesn't have _opcode mdethod"
                                  .format(type(self)))


class Plus(BinOp):
    """Represents the expression A + B"""

    def __repr__(self):
        return "Plus({},{})".format(repr(self.left), repr(self.right))

    def __str__(self):
        """Print fully parenthesized"""
        return "({} + {})".format(self.left, self.right)

    def _apply(self, left: Real, right: Real) -> Real:
        """Addition of two numeric values (Const nodes)"""
        return left + right

    def _opcode(self):
        return "ADD"


class Minus(BinOp):
    """Represents the expression A - B"""

    def __repr__(self):
        return "Minus({},{})".format(repr(self.left), repr(self.right))

    def __str__(self):
        """Print fully parenthesized"""
        return "({} - {})".format(self.left, self.right)

    def _apply(self, left: Real, right: Real) -> Real:
        """Subtraction of two integers"""
        return left - right

    def _opcode(self):
        return "SUB"


class Times(BinOp):
    """Represents the expression A * B"""

    # __init__ is inherited from BinOp

    def __repr__(self):
        return "Times({},{})".format(repr(self.left), repr(self.right))

    def __str__(self):
        """Print fully parenthesized"""
        return "({} * {})".format(self.left, self.right)

    def _apply(self, left: Real, right: Real):
        """Addition of two numeric values"""
        return left * right

    def _opcode(self):
        return "MUL"


class Div(BinOp):
    """Truncating division of two numeric values.
    Note this differs from our original calculator!
    """

    def __repr__(self):
        return "Div({},{})".format(repr(self.left), repr(self.right))

    def __str__(self):
        """Print fully parenthesized"""
        return "({}/{})".format(self.left, self.right)

    def _apply(self, left: Real, right: Real):
        """Addition of two numeric values (Const nodes)"""
        return left // right

    def _opcode(self):
        return "DIV"


class UnOp(Expr):
    """Abstract superclass for unary expressions like negation"""

    def __init__(self, left: Expr):
        """A unary operation has only a left  sub-expression"""
        assert isinstance(left, Expr)
        self.left = left

    def __eq__(self, other):
        """Identical expression"""
        return type(self) == type(other) \
            and self.left == other.left

    def eval(self, env: Env) -> Const:
        """Evaluation strategy for unary expressions"""
        log.debug("Evaluating {} in UnOp".format(self))
        lval = self.left.eval(env)
        assert isinstance(lval, Const), "Op {} applies to numbers, not to {}".format(
            type(self).__name__, lval)
        lval_n = lval.value()
        return Const(self._apply(lval_n))

    def gen(self, contex: Context, target: str):
        """Each unary operator should provide a code generation method"""
        raise NotImplementedError("Unary operator class {} did not implement gen method".format(type(self).__name__))

    def _apply(self, val: int) -> int:
        raise NotImplementedError("Class {} has not implemented _apply".format(
            type(self).__name__))


class Neg(UnOp):
    """Numeric negation"""

    def _apply(self, val: Real) -> Real:
        """Negation of a numeric value"""
        return 0 - val

    def __repr__(self):
        return "Neg({})".format(repr(self.left))

    def __str__(self):
        """Print fully parenthesized"""
        return "~{}".format(self.left)

    def gen(self, context: Context, target: str):
        """Code generation for negation, implemented by 
        subtracting from zero. 
        """
        self.left.gen(context, target)
        context.add_line("\tSUB  {},r0,{}".format(target, target))
        return
