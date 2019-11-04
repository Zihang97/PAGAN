import yaml

def load_hparam(filename):
    f = open(filename, 'r')
    docc = dict()
    docs = yaml.load_all(f)
    for doc in docs:
        docc.update(doc)
    return docc

class Dotdict(dict):
    # __getitem__等四个是python魔术方法，就是在对类的键值操作时会被这三个函数拦截
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            # hasattr:判断对象是否有相关属性
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value


class Hparam(Dotdict):

    def __init__(self, file='config/config.yaml'):
        super(Dotdict, self).__init__()
        # 加载数据到字典
        hp_dict = load_hparam(file)
        # 将字典的key加value的属性
        hp_dotdict = Dotdict(hp_dict)
        # 因为init中不能return，这行起到类似输出的作用
        for k, v in hp_dotdict.items():
            setattr(self, k, v)
    #
    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__

    
hparam = Hparam()
