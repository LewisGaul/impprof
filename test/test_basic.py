import impprof


def test_pkg1():
    import argparse

    with impprof.Monitor() as monitor:
        import collections
        import pkg1

    imps = monitor.imports
    assert set(imps.keys()).intersection({"pkg1", "argparse", "collections"})
