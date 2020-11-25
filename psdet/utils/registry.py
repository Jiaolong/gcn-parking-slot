import inspect

class Registry(object):
    def __init__(self, name):
        self.name = name
        self.obj_dict = {}

    def get(self, key):
        if not key in self.obj_dict:
            raise ValueError("{} is not a registered class.".format(key))
        return self.obj_dict.get(key, None)

    def register(self, cls):
        if not inspect.isclass(cls):
            raise TypeError('module must be a class, but got {}'.format(cls))
        cls_name = cls.__name__
        if cls_name in self.obj_dict:
            raise KeyError("{} is already registered in {}".format(cls_name, self.name))

        self.obj_dict[cls_name] = cls 
        return cls

def build_from_cfg(cfg, registry, default_args=None):
    """
    Build an object from config dict.
    """
    obj_type = cfg.pop('type')
    obj_class = registry.get(obj_type)
    if default_args is not None:
        return obj_class(cfg=cfg, **default_args)
    else:
        return obj_class(cfg=cfg)
