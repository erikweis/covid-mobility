def get_object_params(obj,params_to_ignore=None):

    """Get all attributes of an object.

    Returns:
        [dictionary]: Dict of each attribute of the object and its value
    """

    if params_to_ignore:
        return {attr:val for attr, val in obj.__dict__.items() if attr not in params_to_ignore\
            and not attr.startswith('_')}
    else:
        return obj.__dict__
    