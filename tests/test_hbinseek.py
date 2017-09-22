from __future__ import unicode_literals


class Dummy(object):
    pass


def test_hbinseek(tmpdir):
    import hbinseek
    import os
    import numpy
    import pytest

    filename = os.path.join(str(tmpdir), 'check.hbinseek')
    with hbinseek.open(filename, 'w') as f:
        group = f.create_group('/A/B')
        assert group.name == 'B'
        assert group.path == '/A/B'
        arr = numpy.array([1, 2, 3], dtype=numpy.int16)
        arr2 = numpy.array([[1, 2], [3, 4]], dtype=numpy.int32)

        group.create_array('arr1', arr)
        group.create_array('arr2', arr2)
        group.set_attr('attr1', 'attr1')
        group.set_attr('attr1', 1)
        group.set_attr('attr2', 2)

        with pytest.raises(ValueError):
            group.set_attr('attr1', Dummy())

        with pytest.raises(ValueError):
            group.set_attr('attr1', numpy.array([1, 2, 3]))

        assert set(group.list_attrs()) == {'attr1', 'attr2'}
        groups = f.list_groups()
        assert len(groups) == 1
        assert len(f) == 2
        array = group.get_array('arr1')
        assert array.name == 'arr1'
        assert array.record_offset == 14
        assert array.data_offset == 53
        assert array.dtype == numpy.int16

    # print(open(filename).read())
    # print(open(filename + '.binseekbin').read())

    with hbinseek.open(filename, 'r') as f:
        groups = f.list_groups()
        assert len(groups) == 1
        assert len(f) == 2

        assert f['/A'].list_attrs() == []
        groupb = f['/A/B']
        assert set(groupb.list_attrs()) == {'attr1', 'attr2'}
        assert groupb.get_attr('attr1') == 1
        assert groupb.get_attr('attr2') == 2
        array = groupb.get_array('arr1')
        assert array.name == 'arr1'
        assert array.record_offset == 14
        assert array.data_offset == 53
        assert array.dtype == numpy.int16

        read = array.read_numpy()
        assert numpy.array_equal(read, arr)

        array = groupb.get_array('arr2')
        assert array.name == 'arr2'
        assert array.record_offset == 59
        assert array.data_offset == 102
        assert array.dtype == numpy.int32

        read = array.read_numpy()
        assert numpy.array_equal(read, arr2)
