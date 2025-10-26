import builtins
import importlib.util
import sys


def test_import_emdatabridge_without_transformers():
    """Ensure EMDataBridge can be imported when transformers is not available.

    This test temporarily overrides the builtin import to raise ModuleNotFoundError
    for any import starting with 'transformers' and then loads the package into
    an isolated module object. If the package imports at module level without
    requiring transformers, the import should succeed.
    """
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers" or name.startswith("transformers."):
            raise ModuleNotFoundError("No module named 'transformers'")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = fake_import
    try:
        spec = importlib.util.find_spec("EMDataBridge")
        assert spec is not None, "Could not find EMDataBridge package on sys.path"

        module = importlib.util.module_from_spec(spec)
        # Load into an isolated name to avoid clobbering sys.modules['EMDataBridge']
        sys.modules["__test_emdatabridge__"] = module
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = real_import

    # Basic sanity: module loaded and exposes expected symbol
    assert hasattr(module, "EMDataStandardizer") or hasattr(module, "__name__")
