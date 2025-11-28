# compiled_pyke_files.py

from pyke import target_pkg

pyke_version = '1.1.1'
compiler_version = 1
target_pkg_version = 1

try:
    loader = __loader__
except NameError:
    loader = None

def get_target_pkg():
    return target_pkg.target_pkg(__name__, __file__, pyke_version, loader, {
         ('models', 'symbolic_solvers/pyke_solver/.cache_program/', 'facts.kfb'):
           [1764346147.6767766, 'facts.fbc'],
         ('models', 'symbolic_solvers/pyke_solver/.cache_program/', 'rules.krb'):
           [1764346147.6910694, 'rules_fc.py'],
        },
        compiler_version)

