'''
hbinseek is a library which proposes a file format for dealing with binary data.

The format is a hierarchical format which defines a hierarchy of groups with metadata (much like the
idea behind XML), but with a focus on binary data.

All data is organized in 3 files, one file with the metadata, another with the actual binary data and the
last one a log (which can be used to regenerate the metadata, much like an actual database).

The design is so that adding data to the file is extremely fast (i.e.: binary data is appended to the
binary file and the related info is added to the log file and metadata -- it's also possible to
just skip writing the metadata completely and just recreate it from the log later on so that opening
on a write is even faster -- reading it later on to read data may be a bit slower to recompute the
needed data, but it can benefit scenarios such as a simulator which just dumps information
from time to time).

The design also allows opening a file for reading while another file writes it and working with
incomplete data (everything written up to a point should be consistent).

Format is as follows:

metadata: json file

log file: records file

binary data: arrays dumped

API notes:
All names (group, array) should be received/handled as unicode and saved as utf-8 internally.


Current state:

    - Save/load numpy array.
    - Provides metadata in separated json file.
    
Future work:

    - Restore metadata from log file.
    - Compress array to write and uncompress to read (provide chunks).
    - Document format and provide a C API for the format.
    - Write a custom numpy structure (i.e.: table) and provide a way to retrieve one of its columns.

Related:

https://github.com/fabioz/zarr/blob/master/zarr/
https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
'''

from __future__ import unicode_literals

import sys
import weakref

import numpy

PY3_ONWARDS = sys.version_info[0] >= 3

if PY3_ONWARDS:
    VALID_ATTR_TYPES = (int, float, str, bytes)
    file = open
else:
    VALID_ATTR_TYPES = (int, long, float, unicode, bytes)


class _Array(object):

    def __init__(self, hbinseek, array_name, array_record_offset, data_offset, dtype, shape, order, len_in_bytes):
        if dtype.hasobject:
            raise RuntimeError('Cannot deal with custom Python types.')
        self._hbinseek = hbinseek
        self._name = array_name
        self._record_offset = array_record_offset
        self._data_offset = data_offset
        self._dtype = dtype
        self._shape = shape
        self._order = order
        self._bytes_len = len_in_bytes

        assert order in ('F', 'C')

    @property
    def name(self):
        return self._name

    @property
    def record_offset(self):
        return self._record_offset

    @property
    def data_offset(self):
        return self._data_offset

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def order(self):
        return self._order

    @property
    def bytes_len(self):
        return self._bytes_len

    @classmethod
    def _from_json(cls, hbinseek, json):
        return _Array(
            hbinseek,
            json['name'],
            json['record_offset'],
            json['data_offset'],
            numpy.dtype(json['dtype']),
            json['shape'],
            json['order'],
            json['bytes_len'],
        )

    def _to_json(self):
        return {
            'name': self._name,
            'record_offset': self._record_offset,
            'data_offset': self._data_offset,
            'dtype': self._dtype.str,
            'order': self._order,
            'shape': self._shape,
            'bytes_len': self._bytes_len,
        }

    def read_numpy(self):
        if self._hbinseek.mode != 'r':
            raise RuntimeError('May only read an array in read mode.')

        binary_stream = self._hbinseek._binary_stream
        binary_stream.seek(self._data_offset)
        bytes_read = binary_stream.read(self._bytes_len)
        if len(bytes_read) != self._bytes_len:
            raise IOError(
                'Unable to read required bytes. Binary file seems corrupted.')

        # https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
        # has a better way of doing things, but let's leave it as is for now.
        array = numpy.fromstring(bytes_read, dtype=self._dtype)
        if self._order == 'F':
            array.shape = self._shape[::-1]
            array = array.transpose()
        else:
            array.shape = self._shape

        return array


class _Group(object):

    def __init__(self, parent, hbinseek, group_name, group_path):
        self._hbinseek = hbinseek
        self._name = group_name
        self._attrs = {}
        self._arrays = {}
        self._children_groups = {}
        if parent is None:
            self._parent = lambda: None
        else:
            self._parent = weakref.ref(parent)
        if '//' in group_path:
            raise RuntimeError('Invalid group path: %s' % (group_path,))
        self._path = group_path
        if parent is not None:
            parent._children_groups[group_name] = self

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._name

    def create_array(self, array_name, data):
        array = self._hbinseek._write_array(self._name, array_name, data)
        self._arrays[array_name] = array

    def get_array(self, array_name):
        return self._arrays[array_name]

    def set_attr(self, attr_name, value):
        if not isinstance(value, VALID_ATTR_TYPES):
            raise ValueError('Unexpected type for %s: %s' %
                             (attr_name, type(value),))
        self._attrs[attr_name] = value

    def list_attrs(self):
        if PY3_ONWARDS:
            return list(self._attrs.keys())
        else:
            return self._attrs.keys()

    def get_attr(self, attr_name):
        return self._attrs[attr_name]

    def create_group(self, group_name):
        if '/' in group_name:
            raise ValueError(
                'Not expecting "/" to be in the group name: %s' % (group_name,))

        if self._path.endswith('/'):
            full_path = self._path + group_name
        else:
            full_path = self._path + '/' + group_name

        return self._hbinseek.create_group(full_path)

    def list_groups(self):
        if PY3_ONWARDS:
            return list(self._children_groups.keys())
        else:
            return self._children_groups.keys()

    #----------------------------------------------------------------------------- Group private API

    def _to_json(self):
        ret = {
            "children": [group._to_json() for group in self._children_groups.values()],
            "name": self._name,
            "attrs": self._attrs,
            "arrays": [array._to_json() for array in self._arrays.values()]
        }

        return ret

    def _from_json(self, json):
        children = json['children']
        self._name = json['name']
        self._attrs = json['attrs']
        for array in json['arrays']:
            self._arrays[array['name']] = _Array._from_json(
                self._hbinseek, array)
        for child in children:
            group = self.create_group(child['name'])
            group._from_json(child)


class _HBinseek(object):

    def __init__(self, filename, mode='r', binary_data_filename=None, log_filename=None, autoflush=True):
        '''
        :param mode:
            r = read-only (no write)
            w = write (no read available)
        '''
        if binary_data_filename is None:
            binary_data_filename = filename + '.hbinseekbin'

        if log_filename is None:
            log_filename = filename + '.hbinseeklog'

        if mode not in ('r', 'w'):
            raise ValueError(
                'Expected mode to be "r" or "w". Received: %s' % (mode,))

        self._mode = mode

        if mode == 'r':
            import os
            if not os.path.exists(filename) and not os.path.exists(log_filename):
                raise OSError(
                    'Unable to open for read file which does not exist: %s' % (filename,))

        self._all_groups = {}
        self._root_group = _Group(None, self, '/', '/')
        self._binary_stream = file(binary_data_filename, mode + 'b')
        self._log_stream = file(log_filename, mode + 'b')
        self._autoflush = autoflush
        self._metadata_filename = filename
        self._loaded_info = False

        if self._mode == 'r':
            self._load_metadata()

        if self._mode == 'w':
            # Header: filetype and version
            self._binary_stream.write(b'%BINSEEK v001\n')

    @property
    def mode(self):
        return self._mode

    def _load_metadata(self):
        import json
        import os
        import io
        if not self._loaded_info:
            if os.path.exists(self._metadata_filename):
                with io.open(self._metadata_filename, 'r', encoding='utf-8') as f:
                    loaded_json = json.load(f)
                    self._root_group._from_json(loaded_json)

            self._loaded_info = True

    def list_groups(self):
        return self._root_group.list_groups()

    def all_groups(self):
        if PY3_ONWARDS:
            return list(self._all_groups.values())
        else:
            return self._all_groups.values()

    def __getitem__(self, group_path):
        return self._all_groups[group_path]

    def __len__(self):
        return len(self._all_groups)

    def _check_writable(self):
        if self._mode != 'w':
            raise RuntimeError(
                'Unable to write when not in write mode. Current mode: %s' % (self._mode,))

    def create_array(self, group_name, array_name, data):
        self.create_group(group_name).create_array(array_name, data)

    def create_group(self, group_name):
        if not group_name:
            raise ValueError('Group name must be given.')

        assert group_name.startswith('/')
        group_name = group_name[1:]
        full_path = '/'
        parent = self._root_group
        for group_part in group_name.split('/'):
            full_path += group_part
            try:
                group = self._all_groups[full_path]
            except KeyError:
                group = self._all_groups.setdefault(
                    full_path, _Group(parent, self, group_part, full_path))

            parent = group
            full_path += '/'
        return group

    def _write_array(self, group_name, array_name, data):
        import struct
        self._check_writable()

        if PY3_ONWARDS:
            assert type(array_name) == str
            assert type(group_name) == str

        else:
            assert type(array_name) == bytes
            assert type(group_name) == bytes

        array_name_bytes = array_name.encode('utf-8')

        # Internally always save in C order.
        data_as_bytes = data.tobytes('C')

        binary_stream = self._binary_stream
        log_stream = self._log_stream

        curr_offset = binary_stream.tell()

        to_log = []
        to_log.append(b'ARRAY:')

        # long with array name size and array size
        to_log.append(struct.pack('l', len(array_name_bytes)))
        to_log.append(array_name_bytes)
        to_log.append(b':')

        # long (number of bytes of the data)
        to_log.append(struct.pack('l', len(data_as_bytes)))
        to_log.append(b':')

        dtype_bytes = data.dtype.str.encode('ascii')
        to_log.append(struct.pack('l', len(dtype_bytes)))
        to_log.append(dtype_bytes)
        to_log.append(b':')

        to_log.append(struct.pack('l', len(data.shape)))
        for i in data.shape:
            to_log.append(struct.pack('l', i))
        to_log.append(b':')

        # C or Fortran order
        order = b'C'
        if data.flags.f_contiguous:
            order = b'F'
        to_log.append(b':')
        to_log.append(order)

        # offset to record start
        record_start = b''.join(to_log)

        log_stream.write(record_start)
        log_stream.write(struct.pack('l', curr_offset))

        # end record
        log_stream.write(b'\n')

        if self._autoflush:
            log_stream.flush()

        binary_stream.write(record_start)
        binary_stream.write(data_as_bytes)

        if self._autoflush:
            binary_stream.flush()

        return _Array(
            self,
            array_name,
            curr_offset,
            curr_offset + len(record_start),
            data.dtype,
            data.shape,
            order.decode('ascii'),
            len(data_as_bytes),
        )

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self):
        if self._mode == 'w':
            metadata = self._root_group._to_json()

            import json
            dumped = json.dumps(metadata)
            if not isinstance(dumped, bytes):
                dumped = dumped.encode('utf-8')

            with file(self._metadata_filename, 'wb') as metadata_stream:
                metadata_stream.write(dumped)

        self._binary_stream.close()
        self._log_stream.close()


def open(filename, mode='r', binary_data_filename=None, log_filename=None):
    '''
    :param str filename:
        The name of the metadata file to be handled.

    :param str mode:
        The mode to open the file (w or r).
    '''
    return _HBinseek(filename, mode, binary_data_filename, log_filename)
