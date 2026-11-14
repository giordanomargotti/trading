"""Caricamento controllato dei pickle del repository backtest_put_credit_spread.
Un pickle può eseguire codice arbitrario: si ammettono solo i ricostruttori di pandas/numpy
e builtins.slice, verificati prima con pickletools."""
import pickle, sys, importlib
import numpy as np, pandas as pd

AMMESSI = ("pandas.", "numpy.")
class UnpicklerRistretto(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "builtins" and name == "slice":
            return slice
        if module.startswith(AMMESSI) or module in ("pandas", "numpy"):
            if module.startswith("numpy._core") and not hasattr(np, "_core"):
                module = module.replace("numpy._core", "numpy.core")
            return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f"classe non ammessa: {module}.{name}")

def carica(percorso):
    with open(percorso, "rb") as f:
        return UnpicklerRistretto(f).load()

if __name__ == "__main__":
    for p in sys.argv[1:]:
        obj = carica(p)
        print("===", p.split("/")[-1], type(obj).__name__, list(obj.keys()) if isinstance(obj, dict) else "")
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, pd.DataFrame):
                    print(f"  [{k}] {v.shape}  colonne: {list(v.columns)}")
                    for c in ("entry_date", "date", "expire_date"):
                        if c in v.columns:
                            print(f"     {c}: {v[c].min()} -> {v[c].max()}")
                else:
                    print(f"  [{k}] {type(v).__name__}: {str(v)[:200]}")
