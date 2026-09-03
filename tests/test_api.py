"""The frozen interface surface, and whether it still tells the truth.

`include/vrgrid/api.py` declares the contract and does not serve it: every
function raises. That is the design. What is NOT the design is the message
going stale, and it had -- each one named a Day-1/Day-2 OWNER rather than a
destination, so once the work landed all nine read as outstanding work on
contracts that had been met for days.

`grid/fusion.py` carries the same lesson at the bottom of it: a §10.4 stub
sat beside the real implementation long enough for a second reader to find
the stub first and conclude the ghost gate was unbuilt.

So the pointers are asserted, not trusted. An owner tag rots silently; a
dotted path that has to resolve does not.
"""

import importlib

import pytest
from vrgrid import api

# Every function the frozen surface declares, and where the contract is met.
# `export_gridmap` is deliberately absent -- see the test below.
IMPLEMENTED = {
    api.scatter: "vrgrid.grid.fusion.scatter",
    api.fuse: "vrgrid.grid.fusion.fuse",
    api.split: "vrgrid.grid.splitmerge.split",
    api.merge: "vrgrid.grid.splitmerge.merge",
    api.query: "vrgrid.grid.query.query",
    api.is_traversable: "vrgrid.grid.query.is_traversable",
    api.query_conservative: "vrgrid.gpu.pyramid.classify",
    api.dynamic_objects: "vrgrid.grid.transient.TrackList",
}


def _resolve(dotted):
    module, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("dotted", sorted(set(IMPLEMENTED.values())))
def test_every_destination_the_frozen_api_names_actually_exists(dotted):
    """The pointer resolves. This is the whole point of the file: a message
    that names a module and an attribute fails loudly when either is renamed,
    where "Aakash, Day 2" stays green forever."""
    assert callable(_resolve(dotted))


@pytest.mark.parametrize("stub,dotted", list(IMPLEMENTED.items()),
                         ids=[f.__name__ for f in IMPLEMENTED])
def test_each_stub_raises_pointing_at_its_implementation(stub, dotted):
    """Calling a declaration is an error, and the error says where to go.

    The signatures are frozen and take no map handle, so none of these can be
    made to forward -- that would need module-level state and is a change to
    the surface, which is a room decision. Pointing is what this file can
    honestly do.
    """
    n_args = stub.__code__.co_argcount
    with pytest.raises(NotImplementedError) as excinfo:
        stub(*([None] * n_args))

    message = str(excinfo.value)
    assert "implemented:" in message, f"{stub.__name__} does not name a destination"
    assert dotted in message, f"{stub.__name__} points at {message!r}, not {dotted}"


def test_export_gridmap_is_the_one_that_is_still_open():
    """The negative control, and the reason this file is not just a rename.

    If every stub said "implemented", the assertion above would be satisfied by
    a blanket edit rather than by the work existing. `adapters/` holds nothing
    but `__init__.py`, so this contract really is unserved and has to keep
    reading that way.
    """
    with pytest.raises(NotImplementedError) as excinfo:
        api.export_gridmap()
    assert "implemented:" not in str(excinfo.value)
    assert "optional" in str(excinfo.value)


def test_the_frozen_surface_is_still_the_five_plus_the_output_interface():
    """Names and arities, pinned. Master v4 §3.7 froze these on Day 0 and
    changing one requires all three devs in the same room -- so a diff that
    moves an argument should turn this red before it reaches review."""
    assert api.scatter.__code__.co_argcount == 3          # points, labels, pose
    assert api.fuse.__code__.co_argcount == 0
    assert api.split.__code__.co_argcount == 1            # cell_index
    assert api.merge.__code__.co_argcount == 1            # child_indices
    assert api.query.__code__.co_argcount == 2            # x, y
    assert api.is_traversable.__code__.co_argcount == 2
    assert api.query_conservative.__code__.co_argcount == 1
    assert api.dynamic_objects.__code__.co_argcount == 0
    assert api.export_gridmap.__code__.co_argcount == 0
