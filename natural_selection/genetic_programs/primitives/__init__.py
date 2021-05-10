class Operator:

    operator_label : str = ''

    def __init__(self, operator_label : str = None, function = None):
        self.operator_label = operator_label
        self.function = function

    def exec(self, args):
        return self.function(args)

class OperatorAdd(Operator):

    def __init__(self, operator_label: str = 'add', function=None):
        super().__init__(operator_label, function)

    def exec(self, args):
        a = 0
        for v in args:
            a += v
        return a


class OperatorSub(Operator):

    def __init__(self, operator_label: str = 'sub', function=None):
        super().__init__(operator_label, function)

    def exec(self, args):
        a = args[0]
        for v in args[1:]:
            a -= v
        return a


class OperatorMul(Operator):

    def __init__(self, operator_label: str = 'mul', function=None):
        super().__init__(operator_label, function)

    def exec(self, args):
        a = args[0]
        for v in args[1:]:
            a = a * v
        return a

class OperatorDiv(Operator):

    def __init__(self, operator_label: str = 'div', function=None):
        super().__init__(operator_label, function)

    def exec(self, args):
        a = args[0]
        for v in args[1:]:
            a = a / v
        return a
