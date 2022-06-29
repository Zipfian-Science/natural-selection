class Operator:
    """
    Default class for creating custom operators. The class can either be extended or a custom function can be given.

    Args:
        operator_label (str): The label used for node representation (default = '').
        function (Callable): A custom function used for performing the execution (default = None).
        min_arity (int): The minimum arity of the function (default = 0).
        max_arity (int): The maximum arity of the function (default = 10).
        strict_precedence (bool): Whether the order of the arguments are strictly required (default = True).
    """

    # TODO define strongly typed sets

    def __init__(self, operator_label : str = '', function = None, min_arity : int = 0, max_arity : int = 10, strict_precedence : bool = True):
        self.operator_label = operator_label
        self.function = function
        self.max_arity = max_arity
        self.min_arity = min_arity
        self.strict_precedence = strict_precedence

    def exec(self, args):
        """
        Takes a list of values (arguments) and performs the given or defined function, then returns the result.

        Args:
            args (list): The list of ordered arguments for the function.

        Raises:
            AssertionError: If the length of the list of args is not in the range of the min and max arity.

        Returns:
            result: The resulting value.
        """
        if len(args) > self.max_arity or len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")
        return self.function(args)

class OperatorReturn(Operator):

    def __init__(self, operator_label: str = 'return'):
        """
        Merely returns the given argument.

        Args:
            operator_label (str): The label used for node representation (default = 'return').
        """
        super().__init__(operator_label=operator_label, min_arity=1, max_arity=1)

    def exec(self, args):
        """
        Takes a list of length 1 and returns the value.

        Args:
            args (list): The list of ordered arguments for the function.

        Raises:
            AssertionError: If the length of the list of args is not in the range of the min and max arity.

        Returns:
            result: The resulting value.
        """
        if len(args) > self.max_arity or len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")
        return args[0]

class OperatorAdd(Operator):
    """
    Simple addition of arguments.

    Args:
        operator_label (str): The label used for node representation (default = 'add').
        max_arity (int): The maximum arity of the function (default = 2).
    """
    def __init__(self, operator_label: str = 'add', max_arity : int = 2):
        super().__init__(operator_label=operator_label, min_arity=2, max_arity=max_arity, strict_precedence=False)

    def exec(self, args):
        """
        Takes a list of values (arguments) and performs addition, then returns the result.

        Args:
            args (list): The list of ordered arguments for the function.

        Raises:
            AssertionError: If the length of the list of args is not in the range of the min and max arity.

        Returns:
            result: The resulting value.
        """
        if len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")
        a = 0
        for v in args:
            a += v
        return a

class OperatorSub(Operator):
    """
    Simple subtraction of arguments.

    Args:
        operator_label (str): The label used for node representation (default = 'sub').
        max_arity (int): The maximum arity of the function (default = 2).
    """
    def __init__(self, operator_label: str = 'sub', max_arity : int = 2):
        super().__init__(operator_label=operator_label, min_arity=2, max_arity=max_arity, strict_precedence=True)

    def exec(self, args):
        """
        Takes a list of values (arguments) and performs subtraction, then returns the result.

        Args:
            args (list): The list of ordered arguments for the function.

        Raises:
            AssertionError: If the length of the list of args is not in the range of the min and max arity.

        Returns:
            result: The resulting value.
        """
        if len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")
        a = args[0]
        for v in args[1:]:
            a -= v
        return a

class OperatorMul(Operator):
    """
    Simple multiplication of arguments.

    Args:
        operator_label (str): The label used for node representation (default = 'mul').
        max_arity (int): The maximum arity of the function (default = 2).
    """
    def __init__(self, operator_label: str = 'mul', max_arity : int = 2):
        super().__init__(operator_label=operator_label, min_arity=2, max_arity=max_arity, strict_precedence=False)

    def exec(self, args):
        """
        Takes a list of values (arguments) and performs multiplication, then returns the result.

        Args:
            args (list): The list of ordered arguments for the function.

        Raises:
            AssertionError: If the length of the list of args is not in the range of the min and max arity.

        Returns:
            result: The resulting value.
        """
        if len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")
        a = args[0]
        for v in args[1:]:
            a = a * v
        return a

class OperatorDiv(Operator):
    """
    Simple division of arguments.

    Args:
        operator_label (str): The label used for node representation (default = 'div').
        max_arity (int): The maximum arity of the function (default = 2).
    """
    def __init__(self, operator_label: str = 'div', max_arity : int = 2):
        super().__init__(operator_label=operator_label, min_arity=2, max_arity=max_arity, strict_precedence=True)

    def exec(self, args):
        """
        Takes a list of values (arguments) and performs division, then returns the result.

        Args:
            args (list): The list of ordered arguments for the function.

        Raises:
            AssertionError: If the length of the list of args is not in the range of the min and max arity.

        Returns:
            result: The resulting value.
        """
        if len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")
        a = args[0]
        for v in args[1:]:
            a = a / v
        return a

class OperatorPow(Operator):
    """
    Simple power of arguments.

    Args:
        operator_label (str): The label used for node representation (default = 'pow').
        max_arity (int): The maximum arity of the function (default = 2).
    """
    def __init__(self, operator_label: str = 'pow', max_arity : int = 2):
        super().__init__(operator_label=operator_label, min_arity=2, max_arity=max_arity, strict_precedence=True)

    def exec(self, args):
        """
        Takes a list of values (arguments) and performs powering, then returns the result.

        Args:
            args (list): The list of ordered arguments for the function.

        Raises:
            AssertionError: If the length of the list of args is not in the range of the min and max arity.

        Returns:
            result: The resulting value.
        """
        if len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")
        a = args[0]
        for v in args[1:]:
            a = a ** v
        return a

class OperatorLTE(Operator):
    """
    Simple logical operation, less than or equal.

    Args:
        operator_label (str): The label used for node representation (default = 'lte').
    """
    def __init__(self, operator_label: str = 'lte'):
        super().__init__(operator_label=operator_label, min_arity=2, max_arity=2, strict_precedence=True)

    def exec(self, args):
        """
        Takes a list of values (arguments) and tests if the left most is less than or equal to the right.

        Args:
            args (list): The list of ordered arguments for the function.

        Raises:
            AssertionError: If the length of the list of args is not in the range of the min and max arity.

        Returns:
            bool: The resulting value.
        """
        if len(args) > self.max_arity or len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")

        return args[0] <= args[1]

class OperatorLT(Operator):
    """
    Simple logical operation, less than.

    Args:
        operator_label (str): The label used for node representation (default = 'lt').
    """
    def __init__(self, operator_label: str = 'lt'):
        super().__init__(operator_label=operator_label, min_arity=2, max_arity=2, strict_precedence=True)

    def exec(self, args):
        """
        Takes a list of values (arguments) and tests if the left most is less than to the right.

        Args:
            args (list): The list of ordered arguments for the function.

        Raises:
            AssertionError: If the length of the list of args is not in the range of the min and max arity.

        Returns:
            bool: The resulting value.
        """
        if len(args) > self.max_arity or len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")

        return args[0] < args[1]

class OperatorGTE(Operator):
    """
    Simple logical operation, greater than or equal.

    Args:
        operator_label (str): The label used for node representation (default = 'gte').
    """
    def __init__(self, operator_label: str = 'gte'):
        super().__init__(operator_label=operator_label, min_arity=2, max_arity=2, strict_precedence=True)

    def exec(self, args):
        """
        Takes a list of values (arguments) and tests if the left most is greater than or equal to the right.

        Args:
            args (list): The list of ordered arguments for the function.

        Raises:
            AssertionError: If the length of the list of args is not in the range of the min and max arity.

        Returns:
            bool: The resulting value.
        """
        if len(args) > self.max_arity or len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")

        return args[0] >= args[1]

class OperatorGT(Operator):
    """
    Simple logical operation, greater than or equal.

    Args:
        operator_label (str): The label used for node representation (default = 'gt').
    """
    def __init__(self, operator_label: str = 'gt'):
        super().__init__(operator_label=operator_label, min_arity=2, max_arity=2, strict_precedence=True)

    def exec(self, args):
        """
        Takes a list of values (arguments) and tests if the left most is greater than to the right.

        Args:
            args (list): The list of ordered arguments for the function.

        Raises:
            AssertionError: If the length of the list of args is not in the range of the min and max arity.

        Returns:
            bool: The resulting value.
        """
        if len(args) > self.max_arity or len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")

        return args[0] > args[1]

class OperatorEq(Operator):
    """
    Simple logical operation, arguments are equal.

    Args:
        operator_label (str): The label used for node representation (default = 'eq').
    """
    def __init__(self, operator_label: str = 'eq'):
        super().__init__(operator_label=operator_label, min_arity=2, max_arity=2, strict_precedence=False)

    def exec(self, args):
        """
        Takes a list of values (arguments) and tests if the left most is equal to the right.

        Args:
            args (list): The list of ordered arguments for the function.

        Raises:
            AssertionError: If the length of the list of args is not in the range of the min and max arity.

        Returns:
            bool: The resulting value.
        """
        if len(args) > self.max_arity or len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")

        return args[0] == args[1]

class OperatorNE(Operator):
    """
    Simple logical operation, arguments are not equal.

    Args:
        operator_label (str): The label used for node representation (default = 'ne').
    """
    def __init__(self, operator_label: str = 'ne'):
        super().__init__(operator_label=operator_label, min_arity=2, max_arity=2, strict_precedence=False)

    def exec(self, args):
        """
        Takes a list of values (arguments) and tests if the left most is not equal to the right.

        Args:
            args (list): The list of ordered arguments for the function.

        Raises:
            AssertionError: If the length of the list of args is not in the range of the min and max arity.

        Returns:
            bool: The resulting value.
        """
        if len(args) > self.max_arity or len(args) < self.min_arity:
            raise AssertionError(
                f"Number of arguments do not match the allowed arity min:{self.min_arity} max:{self.max_arity}")

        return args[0] != args[1]