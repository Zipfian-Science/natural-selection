class Operator:

    def __init__(self, operator_label : str = '', function = None, min_arity : int = -1, max_arity : int = -1):
        self.operator_label = operator_label
        self.function = function
        self.max_arity = max_arity
        self.min_arity = min_arity

    def exec(self, args):
        return self.function(args)

class OperatorReturn(Operator):

    def __init__(self, operator_label: str = 'return', function = None):
        super().__init__(operator_label, function, min_arity=1, max_arity=1)

    def exec(self, args):
        if len(args) > self.max_arity or len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")
        return args[0]

class OperatorAdd(Operator):

    def __init__(self, operator_label: str = 'add', function = None):
        super().__init__(operator_label, function)

    def exec(self, args):
        a = 0
        for v in args:
            a += v
        return a

class OperatorSub(Operator):

    def __init__(self, operator_label: str = 'sub', function = None):
        super().__init__(operator_label, function)

    def exec(self, args):
        a = args[0]
        for v in args[1:]:
            a -= v
        return a

class OperatorMul(Operator):

    def __init__(self, operator_label: str = 'mul', function = None):
        super().__init__(operator_label, function)

    def exec(self, args):
        a = args[0]
        for v in args[1:]:
            a = a * v
        return a

class OperatorDiv(Operator):

    def __init__(self, operator_label: str = 'div', function = None):
        super().__init__(operator_label, function)

    def exec(self, args):
        a = args[0]
        for v in args[1:]:
            a = a / v
        return a

class OperatorPow(Operator):

    def __init__(self, operator_label: str = 'pow', function = None):
        super().__init__(operator_label, function)

    def exec(self, args):
        a = args[0]
        for v in args[1:]:
            a = a ** v
        return a

class OperatorLTE(Operator):

    def __init__(self, operator_label: str = 'lte', function = None):
        super().__init__(operator_label, function, min_arity=2, max_arity=2)

    def exec(self, args):
        if len(args) > self.max_arity or len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")

        return args[0] <= args[1]

class OperatorLT(Operator):

    def __init__(self, operator_label: str = 'lt', function=None):
        super().__init__(operator_label, function, min_arity=2, max_arity=2)

    def exec(self, args):
        if len(args) > self.max_arity or len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")

        return args[0] < args[1]

class OperatorGTE(Operator):

    def __init__(self, operator_label: str = 'gte', function=None):
        super().__init__(operator_label, function, min_arity=2, max_arity=2)

    def exec(self, args):
        if len(args) > self.max_arity or len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")

        return args[0] >= args[1]

class OperatorGT(Operator):

    def __init__(self, operator_label: str = 'gt', function=None):
        super().__init__(operator_label, function, min_arity=2, max_arity=2)

    def exec(self, args):
        if len(args) > self.max_arity or len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")

        return args[0] > args[1]

class OperatorEq(Operator):

    def __init__(self, operator_label: str = 'eq', function=None):
        super().__init__(operator_label, function, min_arity=2, max_arity=2)

    def exec(self, args):
        if len(args) > self.max_arity or len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")

        return args[0] == args[1]

class OperatorNE(Operator):

    def __init__(self, operator_label: str = 'ne', function=None):
        super().__init__(operator_label, function, min_arity=2, max_arity=2)

    def exec(self, args):
        if len(args) > self.max_arity or len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")

        return args[0] != args[1]