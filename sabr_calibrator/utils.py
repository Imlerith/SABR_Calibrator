def get_months_years(x):
    n_months = x * 12
    if n_months >= 12:
        return f"{int(x)}y"
    else:
        return f"{int(n_months)}m"


def lazy_property(func):
    name = '_lazy_' + func.__name__

    @property
    def lazy(self):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            value = func(self)
            setattr(self, name, value)
            return value
    return lazy
