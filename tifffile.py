#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tifffile.py

# Copyright (c) 2008-2013, Christoph Gohlke
# Copyright (c) 2008-2013, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Read and write image data from and to TIFF files.

Image and meta-data can be read from TIFF, BigTIFF, OME-TIFF, STK, LSM, NIH,
ImageJ, FluoView, SEQ and GEL files.
Only a subset of the TIFF specification is supported, mainly uncompressed
and losslessly compressed 2**(0 to 6) bit integer, 16, 32 and 64-bit float,
grayscale and RGB(A) images, which are commonly used in bio-scientific imaging.
Specifically, reading JPEG/CCITT compressed image data or EXIF/IPTC/GPS/XMP
meta-data is not implemented. Only primary info records are read for STK,
FluoView, and NIH image formats.

TIFF, the Tagged Image File Format, is under the control of Adobe Systems.
BigTIFF allows for files greater than 4 GB. STK, LSM, FluoView, SEQ, GEL,
and OME-TIFF, are custom extensions defined by MetaMorph, Carl Zeiss
MicroImaging, Olympus, Media Cybernetics, Molecular Dynamics, and the Open
Microscopy Environment consortium respectively.

For command line usage run ``python tifffile.py --help``

:Author:
  `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:Version: 2013.03.26

Requirements
------------
* `CPython 2.7, 3.2 or 3.3 <http://www.python.org>`_
* `Numpy 1.7 <http://www.numpy.org>`_
* `Matplotlib 1.2 <http://www.matplotlib.org>`_  (optional for plotting)
* `Tifffile.c 2013.01.18 <http://www.lfd.uci.edu/~gohlke/>`_
  (recommended for faster decoding of PackBits and LZW encoded strings)

Notes
-----
The API is not stable yet and might change between revisions.

Tested on little-endian platforms only.

Other Python packages and modules for reading bio-scientific TIFF files:
* `Imread <http://luispedro.org/software/imread>`_
* `PyLibTiff <http://code.google.com/p/pylibtiff>`_
* `SimpleITK <http://www.simpleitk.org>`_
* `PyLSM <https://launchpad.net/pylsm>`_
* `PyMca.TiffIO.py <http://pymca.sourceforge.net/>`_
* `BioImageXD.Readers <http://www.bioimagexd.net/>`_
* `Cellcognition.io <http://cellcognition.org/>`_
* `CellProfiler.bioformats <http://www.cellprofiler.org/>`_

Acknowledgements
----------------
*  Egor Zindy, University of Manchester, for cz_lsm_scan_info specifics.
*  Wim Lewis, for a bug fix and some read_cz_lsm functions.

References
----------
(1) TIFF 6.0 Specification and Supplements. Adobe Systems Incorporated.
    http://partners.adobe.com/public/developer/tiff/
(2) TIFF File Format FAQ. http://www.awaresystems.be/imaging/tiff/faq.html
(3) MetaMorph Stack (STK) Image File Format.
    http://support.meta.moleculardevices.com/docs/t10243.pdf
(4) File Format Description - LSM 5xx Release 2.0.
    http://ibb.gsf.de/homepage/karsten.rodenacker/IDL/Lsmfile.doc
(5) BioFormats. http://www.loci.wisc.edu/ome/formats.html
(6) The OME-TIFF format.
    http://www.openmicroscopy.org/site/support/file-formats/ome-tiff
(7) TiffDecoder.java
    http://rsbweb.nih.gov/ij/developer/source/ij/io/TiffDecoder.java.html
(8) UltraQuant(r) Version 6.0 for Windows Start-Up Guide.
    http://www.ultralum.com/images%20ultralum/pdf/UQStart%20Up%20Guide.pdf

Examples
--------
>>> data = numpy.random.rand(301, 219)
>>> imsave('temp.tif', data)
>>> image = imread('temp.tif')
>>> assert numpy.all(image == data)

>>> tif = TiffFile('test.tif')
>>> images = tif.asarray()
>>> image0 = tif[0].asarray()
>>> for page in tif:
...     for tag in page.tags.values():
...         t = tag.name, tag.value
...     image = page.asarray()
...     if page.is_rgb: pass
...     if page.is_palette:
...         t = page.color_map
...     if page.is_stk:
...         t = page.mm_uic_tags.number_planes
...     if page.is_lsm:
...         t = page.cz_lsm_info
>>> tif.close()

"""

from __future__ import division, print_function

import sys
import os
import re
import glob
import math
import zlib
import time
import struct
import warnings
import datetime
import collections
from fractions import Fraction
from xml.etree import cElementTree as ElementTree

import numpy

__version__ = '2013.03.26'
__docformat__ = 'restructuredtext en'
__all__ = ['imsave', 'imread', 'imshow', 'TiffFile', 'TiffSequence']


def imsave(filename, data, photometric=None, planarconfig=None,
           resolution=None, description=None, software='tifffile.py',
           byteorder=None, bigtiff=False):
    """Write image data to TIFF file.

    Image data are written uncompressed in one stripe per plane.
    Dimensions larger than 2 or 3 (depending on photometric mode and
    planar configuration) are flattened and saved as separate pages.

    Parameters
    ----------
    filename : str
        Name of file to write.
    data : array_like
        Input image. The last dimensions are assumed to be image height,
        width, and samples.
    photometric : {'minisblack', 'miniswhite', 'rgb'}
        The color space of the image data.
        By default this setting is inferred from the data shape.
    planarconfig : {'contig', 'planar'}
        Specifies if samples are stored contiguous or in separate planes.
        By default this setting is inferred from the data shape.
        'contig': last dimension contains samples.
        'planar': third last dimension contains samples.
    resolution : (float, float) or ((int, int), (int, int))
        X and Y resolution in dots per inch as float or rational numbers.
    description : str
        The subject of the image. Saved with the first page only.
    software : str
        Name of the software used to create the image.
        Saved with the first page only.
    byteorder : {'<', '>'}
        The endianness of the data in the file.
        By default this is the system's native byte order.
    bigtiff : bool
        If True the BigTIFF format is used.
        By default the standard TIFF format is used for data less than 2040 MB.

    Examples
    --------
    >>> data = numpy.random.rand(10, 3, 301, 219)
    >>> imsave('temp.tif', data)

    """
    assert(photometric in (None, 'minisblack', 'miniswhite', 'rgb'))
    assert(planarconfig in (None, 'contig', 'planar'))
    assert(byteorder in (None, '<', '>'))

    if byteorder is None:
        byteorder = '<' if sys.byteorder == 'little' else '>'

    data = numpy.asarray(data, dtype=byteorder+data.dtype.char, order='C')
    data_shape = shape = data.shape
    data = numpy.atleast_2d(data)

    if not bigtiff and data.size * data.dtype.itemsize < 2040*2**20:
        bigtiff = False
        offset_size = 4
        tag_size = 12
        numtag_format = 'H'
        offset_format = 'I'
        val_format = '4s'
    else:
        bigtiff = True
        offset_size = 8
        tag_size = 20
        numtag_format = 'Q'
        offset_format = 'Q'
        val_format = '8s'

    # unify shape of data
    samplesperpixel = 1
    extrasamples = 0
    if photometric is None:
        if data.ndim > 2 and (shape[-3] in (3, 4) or shape[-1] in (3, 4)):
            photometric = 'rgb'
        else:
            photometric = 'minisblack'
    if photometric == 'rgb':
        if len(shape) < 3:
            raise ValueError("not a RGB(A) image")
        if planarconfig is None:
            planarconfig = 'planar' if shape[-3] in (3, 4) else 'contig'
        if planarconfig == 'contig':
            if shape[-1] not in (3, 4):
                raise ValueError("not a contiguous RGB(A) image")
            data = data.reshape((-1, 1) + shape[-3:])
            samplesperpixel = shape[-1]
        else:
            if shape[-3] not in (3, 4):
                raise ValueError("not a planar RGB(A) image")
            data = data.reshape((-1, ) + shape[-3:] + (1, ))
            samplesperpixel = shape[-3]
        if samplesperpixel == 4:
            extrasamples = 1
    elif planarconfig and len(shape) > 2:
        if planarconfig == 'contig':
            data = data.reshape((-1, 1) + shape[-3:])
            samplesperpixel = shape[-1]
        else:
            data = data.reshape((-1, ) + shape[-3:] + (1, ))
            samplesperpixel = shape[-3]
        extrasamples = samplesperpixel - 1
    else:
        planarconfig = None
        data = data.reshape((-1, 1) + shape[-2:] + (1, ))

    shape = data.shape  # (pages, planes, height, width, contig samples)

    bytestr = bytes if sys.version[0] == '2' else lambda x: bytes(x, 'ascii')
    tifftypes = {'B': 1, 's': 2, 'H': 3, 'I': 4, '2I': 5, 'b': 6,
                 'h': 8, 'i': 9, 'f': 11, 'd': 12, 'Q': 16, 'q': 17}
    tifftags = {
        'new_subfile_type': 254, 'subfile_type': 255,
        'image_width': 256, 'image_length': 257, 'bits_per_sample': 258,
        'compression': 259, 'photometric': 262, 'fill_order': 266,
        'document_name': 269, 'image_description': 270, 'strip_offsets': 273,
        'orientation': 274, 'samples_per_pixel': 277, 'rows_per_strip': 278,
        'strip_byte_counts': 279, 'x_resolution': 282, 'y_resolution': 283,
        'planar_configuration': 284, 'page_name': 285, 'resolution_unit': 296,
        'software': 305, 'datetime': 306, 'predictor': 317, 'color_map': 320,
        'extra_samples': 338, 'sample_format': 339}

    tags = []
    tag_data = []

    def pack(fmt, *val):
        return struct.pack(byteorder+fmt, *val)

    def tag(name, dtype, number, value, offset=[0]):
        # append tag binary string to tags list
        # append (offset, value as binary string) to tag_data list
        # increment offset by tag_size
        if dtype == 's':
            value = bytestr(value) + b'\0'
            number = len(value)
            value = (value, )
        t = [pack('HH', tifftags[name], tifftypes[dtype]),
             pack(offset_format, number)]
        if len(dtype) > 1:
            number *= int(dtype[:-1])
            dtype = dtype[-1]
        if number == 1:
            if isinstance(value, (tuple, list)):
                value = value[0]
            t.append(pack(val_format, pack(dtype, value)))
        elif struct.calcsize(dtype) * number <= offset_size:
            t.append(pack(val_format, pack(str(number)+dtype, *value)))
        else:
            t.append(pack(offset_format, 0))
            tag_data.append((offset[0] + offset_size + 4,
                             pack(str(number)+dtype, *value)))
        tags.append(b''.join(t))
        offset[0] += tag_size

    def rational(arg, max_denominator=1000000):
        # return nominator and denominator from float or two integers
        try:
            f = Fraction.from_float(arg)
        except TypeError:
            f = Fraction(arg[0], arg[1])
        f = f.limit_denominator(max_denominator)
        return f.numerator, f.denominator

    if software:
        tag('software', 's', 0, software)
    if description:
        tag('image_description', 's', 0, description)
    elif shape != data_shape:
        tag('image_description', 's', 0,
            "shape=(%s)" % (",".join('%i' % i for i in data_shape)))
    tag('datetime', 's', 0,
        datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"))
    # write previous tags only once
    writeonce = (len(tags), len(tag_data)) if shape[0] > 1 else None
    tag('compression', 'H', 1, 1)
    tag('orientation', 'H', 1, 1)
    tag('image_width', 'I', 1, shape[-2])
    tag('image_length', 'I', 1, shape[-3])
    tag('new_subfile_type', 'I', 1, 0 if shape[0] == 1 else 2)
    tag('photometric', 'H', 1,
        {'miniswhite': 0, 'minisblack': 1, 'rgb': 2}[photometric])
    tag('samples_per_pixel', 'H', 1, samplesperpixel)
    if planarconfig:
        tag('planar_configuration', 'H', 1, 1 if planarconfig=='contig' else 2)
        tag('bits_per_sample', 'H', samplesperpixel,
            (data.dtype.itemsize * 8, ) * samplesperpixel)
    else:
        tag('bits_per_sample', 'H', 1, data.dtype.itemsize * 8)
    tag('sample_format', 'H', 1, {'u': 1, 'i': 2, 'f': 3, 'c': 6}[data.dtype.kind])
    if extrasamples:
        if photometric == 'rgb':
            tag('extra_samples', 'H', 1, 1)  # alpha channel
        else:
            tag('extra_samples', 'H', extrasamples, (0, ) * extrasamples)
    if resolution:
        tag('x_resolution', '2I', 1, rational(resolution[0]))
        tag('y_resolution', '2I', 1, rational(resolution[1]))
        tag('resolution_unit', 'H', 1, 2)
    tag('rows_per_strip', 'I', 1, shape[-3])
    # use one strip per plane
    strip_byte_counts = (data[0, 0].size * data.dtype.itemsize, ) * shape[1]
    tag('strip_byte_counts', offset_format, shape[1], strip_byte_counts)
    # strip_offsets must be the last tag; will be updated later
    tag('strip_offsets', offset_format, shape[1], (0, ) * shape[1])

    fh = open(filename, 'wb')
    seek = fh.seek
    tell = fh.tell

    def write(arg, *args):
        fh.write(pack(arg, *args) if args else arg)

    write({'<': b'II', '>': b'MM'}[byteorder])
    if bigtiff:
        write('HHH', 43, 8, 0)
    else:
        write('H', 42)
    ifd_offset = tell()
    write(offset_format, 0)  # first IFD
    for i in range(shape[0]):
        # update pointer at ifd_offset
        pos = tell()
        seek(ifd_offset)
        write(offset_format, pos)
        seek(pos)
        # write tags
        write(numtag_format, len(tags))
        tag_offset = tell()
        write(b''.join(tags))
        ifd_offset = tell()
        write(offset_format, 0)  # offset to next IFD
        # write extra tag data and update pointers
        for off, dat in tag_data:
            pos = tell()
            seek(tag_offset + off)
            write(offset_format, pos)
            seek(pos)
            write(dat)
        # update strip_offsets
        pos = tell()
        if len(strip_byte_counts) == 1:
            seek(ifd_offset - offset_size)
            write(offset_format, pos)
        else:
            seek(pos - offset_size*shape[1])
            strip_offset = pos
            for size in strip_byte_counts:
                write(offset_format, strip_offset)
                strip_offset += size
        seek(pos)
        # write data
        data[i].tofile(fh)  # if this fails try to update Python and numpy
        fh.flush()
        # remove tags that should be written only once
        if writeonce:
            tags = tags[writeonce[0]:]
            d = writeonce[0] * tag_size
            tag_data = [(o-d, v) for (o, v) in tag_data[writeonce[1]:]]
            writeonce = None
    fh.close()


def imread(files, *args, **kwargs):
    """Return image data from TIFF file(s) as numpy array.

    The first image series is returned if no arguments are provided.

    Parameters
    ----------
    files : str or list
        File name, glob pattern, or list of file names.
    key : int, slice, or sequence of page indices
        Defines which pages to return as array.
    series : int
        Defines which series of pages in file to return as array.
    multifile : bool
        If True (default), OME-TIFF data may include pages from multiple files.
    pattern : str
        Regular expression pattern that matches axes names and indices in
        file names.

    Examples
    --------
    >>> im = imread('test.tif', 0)
    >>> im.shape
    (256, 256, 4)
    >>> ims = imread(['test.tif', 'test.tif'])
    >>> ims.shape
    (2, 256, 256, 4)

    """
    kwargs_file = {}
    if 'multifile' in kwargs:
        kwargs_file['multifile'] = kwargs['multifile']
        del kwargs['multifile']
    else:
        kwargs_file['multifile'] = True
    kwargs_seq = {}
    if 'pattern' in kwargs:
        kwargs_seq['pattern'] = kwargs['pattern']
        del kwargs['pattern']

    if isinstance(files, basestring) and any(i in files for i in '?*'):
        files = glob.glob(files)
    if not files:
        raise ValueError('no files found')
    if len(files) == 1:
        files = files[0]

    if isinstance(files, basestring):
        with TiffFile(files, **kwargs_file) as tif:
            return tif.asarray(*args, **kwargs)
    else:
        with TiffSequence(files, **kwargs_seq) as imseq:
            return imseq.asarray(*args, **kwargs)






# -------------------------------------------
# Only the imsave function has been kept
# -------------------------------------------



