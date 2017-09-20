'''
binseek is a library which proposes a file format for dealing with binary data.

The format is a hierarchical format which defines a hierarchy of groups with metadata (much like the
idea behind XML), but with a focus on binary data.

All data is organized in 3 files, one file with the metadata, another with the actual binary data and the
last one a log (which can be used to regenerate the metadata, much like an actual database).

The design is so that adding data to the file is extremely fast (i.e.: binary data is appended to the
file index info is added to the log file and metadata).

The design also allows opening a file for reading while another file writes it.

Format is as follows:

metadata: json file

log file: records file

binary data: arrays dumped

Notes:
Related projects:
http://github.com/fabioz/zarr/blob/master/zarr/
'''

from __future__ import unicode_literals

import sys

PY3_ONWARDS = sys.version_info[0] >= 3

if PY3_ONWARDS:
    VALID_ATTR_TYPES = (int, float, str, bytes)
else:
    VALID_ATTR_TYPES = (int, long, float, unicode, bytes)
    
class _Group(object):

    def __init__(self, binseek, group_name):
        self._binseek = binseek
        self._name = group_name
        self._attrs = {}

    @property
    def name(self):
        return self._name

    def create_array(self, array_name, data):
        self._binseek._write_array(self._name, array_name, data)

    def set_attr(self, attr_name, value):
        if not isinstance(value, VALID_ATTR_TYPES):
            raise ValueError('Unexpected type for %s: %s' % (attr_name, type(value),))
        self._attrs[attr_name] = value

    def list_attrs(self):
        return list(self._attrs.keys())


class _Binseek(object):

    def __init__(self, filename, mode='r', binary_data_filename=None, log_filename=None):
        '''
        :param mode:
            r = read
            r+ = read and write
            w = write
        '''
        if binary_data_filename is None:
            binary_data_filename = filename + '.binseekbin'

        if log_filename is None:
            log_filename = filename + '.binseeklog'

        if mode not in ('r', 'r+', 'w'):
            raise ValueError('Expected mode to be "r", "r+" or "w". Received: %s' % (mode,))

        self._mode = mode
        self._groups = {}
        self._binary_stream = file(filename, mode + 'b')
        self._log_stream = file(filename, mode + 'b')
        self._offset = 0
        self._autoflush = True

    def _check_writable(self):
        if self._mode not in ('r+', 'w'):
            raise RuntimeError('Unable to write when not in write mode. Current mode: %s' % (self._mode,))

    def create_array(self, group_name, array_name, data):
        self.create_group(group_name).create_array(array_name, data)

    def create_group(self, group_name):
        try:
            return self._groups[group_name]
        except KeyError:
            return self._groups.setdefault(group_name, _Group(self, group_name))

    def _write_array(self, group_name, array_name, data):
        self._check_writable()

        if type(array_name) != bytes:
            array_name = array_name.encode('utf-8')

        data_as_bytes = data.tobytes()

        self._log_stream.write(
            b'ARRAY:%s:%s:%s:%s' % (len(array_name), array_name, len(data_as_bytes), self._offset))

        self._binary_stream.write(b'ARRAY:')
        self._binary_stream.write(b'%s' % (len(array_name),))
        self._binary_stream.write(array_name)
        self._binary_stream.write(b'%s' % len(data_as_bytes))
        self._binary_stream.write(data_as_bytes)
        self._offset = self._binary_stream.tell()
        if self._autoflush:
            self._binary_stream.flush()



def open(filename, mode='r', binary_data_filename=None, log_filename=None):
    '''
    :param str filename:
        The name of the metadata file to be handled.

    :param str mode:
        The mode to open the file (w or r).
    '''
    return _Binseek(filename, mode, binary_data_filename, log_filename)
