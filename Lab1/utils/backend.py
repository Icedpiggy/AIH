_use_cupy = False

try:
	import cupy as cp
	if cp.cuda.is_available():
		_use_cupy = True
except (ImportError, Exception):
	pass

import numpy

if _use_cupy:
	np = cp
	from cupyx.scipy import ndimage
	from cupy.lib.stride_tricks import as_strided
else:
	np = numpy
	from scipy import ndimage
	from numpy.lib.stride_tricks import as_strided
