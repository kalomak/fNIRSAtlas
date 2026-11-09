def deep_merge_dicts(a, b):
    """Recursively merge dict b into dict a, returning a new dict."""

    if not b:
        return a

    if not a:
        return b

    result = a.copy()
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def dict2json(data):
    """Convert any dictionary to JSON-serializable format."""
    if isinstance(data, dict):
        return {str(k): dict2json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [dict2json(item) for item in data]
    elif hasattr(data, "dtype"):  # numpy arrays and scalars
        if data.ndim == 0:  # scalar
            return data.item()
        else:  # array
            return data.tolist()
    # elif isinstance(data, (np.integer, np.floating, np.bool_)):
    #     return data.item()
    else:
        return data
