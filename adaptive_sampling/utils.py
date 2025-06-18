def convert_to_nested_dict(flat_dict):
    """
    Converts the flat dictionary with dot-separated neames used in the wandb config into a regular dictionary."""
    nested_dict = {} 
    for key, value in flat_dict.items():
        keys = key.split(".")
        d = nested_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        if value == "True" or value == "False":
            d[keys[-1]] = value == "True"
    return nested_dict