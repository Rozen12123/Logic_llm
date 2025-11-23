import builtins as builtins

_tmp_dict = globals()
_use_kernprof = False


def do_nothing(func):
    return func


if isinstance(_tmp_dict['__builtins__'], dict):
    if 'profile' in _tmp_dict['__builtins__']:
        _use_kernprof = True

if not _use_kernprof:
    builtins.__dict__['profile'] = do_nothing
