SINGLETON_INSTANCES = {}

def singleton(cls):
    global SINGLETON_INSTANCES

    def get_instance(*args, **kwargs):
        if cls not in SINGLETON_INSTANCES:
            SINGLETON_INSTANCES[cls] = cls(*args, **kwargs)
        return SINGLETON_INSTANCES[cls]

    return get_instance