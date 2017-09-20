from __future__ import unicode_literals


class Dummy(object):
    pass

def test_hbinseek(tmpdir):
    import hbinseek
    import os
    import numpy
    import pytest

    f = hbinseek.open(os.path.join(str(tmpdir), 'check.hbinseek'), 'w')
    group = f.create_group('/A/B')
    assert group.name == '/A/B'
    group.create_array('arr1', numpy.array([1, 2, 3]))
    group.set_attr('attr1', 'attr1')
    group.set_attr('attr1', 1)

    with pytest.raises(ValueError):
        group.set_attr('attr1', Dummy())

    with pytest.raises(ValueError):
        group.set_attr('attr1', numpy.array([1, 2, 3]))

    assert group.list_attrs() == ['attr1']
