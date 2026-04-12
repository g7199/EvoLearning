"""Method registry."""
METHOD_REGISTRY = {}


def register_method(cls):
    METHOD_REGISTRY[cls.name] = cls
    return cls


def get_method(name: str):
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[name]


def list_methods():
    return list(METHOD_REGISTRY.keys())
