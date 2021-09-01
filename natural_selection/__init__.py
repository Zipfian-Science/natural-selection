__version__ = '0.2.10'

def get_random_string(length : int = 8, include_numeric=False) -> str:
    """
    Generate a random string with a given length. Used mainly for password generation.
    Args:
        length (int): Length to generate.
        include_numeric (bool): Include numbers?

    Returns:
        str: Random character string.
    """
    import random
    import string

    letters = string.ascii_letters
    if include_numeric:
        letters = f'{letters}{string.digits}'
    return ''.join(random.choice(letters) for i in range(length))