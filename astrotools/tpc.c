#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"
#include <math.h>

typedef struct fast_array_s {
	size_t size;
	double *data;
	void *mem;
} fast_array_t;

int init_fast_array(fast_array_t *fa, PyArrayObject *pa) {
	if (pa == NULL || fa == NULL)
		return 0;
	
	fa->size = PyArray_DIM(pa, 0);
	
	char *data = (char *) PyArray_DATA(pa);
	npy_intp stride = PyArray_STRIDE(pa, 0) ;
	
	fa->mem = malloc(fa->size * sizeof(double) + 15);
	fa->data = (double *) (((uintptr_t) fa->mem + 15) & ~0x0F);
	size_t i;
	for (i = 0; i < fa->size; i++) {
		fa->data[i] = *(double *) (data + i * stride);
	}
	
	return 1;
}

void free_fast_array(fast_array_t *fa) {
	free(fa->mem);
	fa->data = NULL;
	fa->size = 0;
}

void normalize(size_t n, double *x, double *y, double *z) {
	size_t i;
	#pragma omp parallel for private(i), schedule(static, 1000)
	for (i = 0; i < n; i++) {
		double l = 1./sqrt(x[i]*x[i]+y[i]*y[i]+z[i]*z[i]);
		x[i] *= l;
		y[i] *= l;
		z[i] *= l;
	}
}

// Two point autocorrelation
static PyObject *tpac(PyObject *self, PyObject *args) {
	PyArrayObject *ac, *ax, *ay, *az, *aw;
	double maxangle;

	// parse the parameters
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!d", 
		&PyArray_Type, &ac, 
		&PyArray_Type, &ax, 
		&PyArray_Type, &ay, 
		&PyArray_Type, &az, 
		&PyArray_Type, &aw, 
		&maxangle))
		return NULL ;

	fast_array_t x, y, z, w;

	if (!init_fast_array(&x, ax))
		return NULL;
	if (!init_fast_array(&y, ay))
		return NULL;
	if (!init_fast_array(&z, az))
		return NULL;
	if (!init_fast_array(&w, aw))
		return NULL;

	if ((x.size != y.size) || (x.size != z.size) || (x.size != w.size))
		return NULL;
	size_t n = x.size;
	normalize(n, x.data, y.data, z.data);

	// extract parameters and arrays
	npy_intp nbins = PyArray_DIM(ac, 0) ;
	double *ac_data = (double *) PyArray_DATA(ac);
	npy_intp ac_stride = PyArray_STRIDE(ac, 0) ;
	
	size_t i, j, idx;
	double m = cos(maxangle);
	double d = nbins / maxangle;

#pragma omp parallel shared(x, y, z, n, d, nbins) private(i, j, idx)
	{
		// setup temporary result array
		double *tmp = (double *) malloc(nbins * sizeof(double));
		int ib;
		for (ib = 0; ib < nbins; ib++) {
			tmp[ib] = 0.;
		}

		// compute autocorrelation
#pragma omp for schedule(dynamic, 16) nowait
		for (i = 0; i < n - 1; i++) {
			for (j = i + 1; j < n; j++) {
				float cosalpha = x.data[i] * x.data[j] + y.data[i] * y.data[j] + z.data[i] * z.data[j];
				if (cosalpha >= m) {
					idx = acos(cosalpha) * d;
					tmp[idx] += (w.data[i] * w.data[j]);
				}
			}
		}

		// add temporary result to total result
#pragma omp critical
		for (ib = 0; ib < nbins; ib++) {
			ac_data[ib] += tmp[ib];
		}

		// cleanup
		free(tmp);
	}

	free_fast_array(&x);
	free_fast_array(&y);
	free_fast_array(&z);
	free_fast_array(&w);

	Py_INCREF(Py_None);
	return Py_None;
}

// Two point crosscorrelation
static PyObject *tpcc(PyObject *self, PyObject *args) {
	PyArrayObject *ac, *ax1, *ay1, *az1, *aw1, *ax2, *ay2, *az2, *aw2;
	double maxangle;

	// parse the parameters
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!d", 
			&PyArray_Type, &ac, 
			&PyArray_Type, &ax1, 
			&PyArray_Type, &ay1, 
			&PyArray_Type, &az1, 
			&PyArray_Type, &aw1, 
			&PyArray_Type, &ax2, 
			&PyArray_Type, &ay2, 
			&PyArray_Type, &az2, 
			&PyArray_Type, &aw2, 
			&maxangle))
		return NULL ;
	
	fast_array_t x1, y1, z1, w1, x2, y2, z2, w2;

	if (!init_fast_array(&x1, ax1))
		return NULL;
	if (!init_fast_array(&y1, ay1))
		return NULL;
	if (!init_fast_array(&z1, az1))
		return NULL;
	if (!init_fast_array(&w1, aw1))
		return NULL;
	if (!init_fast_array(&x2, ax2))
		return NULL;
	if (!init_fast_array(&y2, ay2))
		return NULL;
	if (!init_fast_array(&z2, az2))
		return NULL;
	if (!init_fast_array(&w2, aw2))
		return NULL;

	if ((x1.size != y1.size) || (x1.size != z1.size) || (x1.size != w1.size))
		return NULL;
	size_t n1 = x1.size;
	normalize(n1, x1.data, y1.data, z1.data);

	if ((x2.size != y2.size) || (x2.size != z2.size) || (x2.size != w2.size))
		return NULL;
	size_t n2 = x2.size;
	normalize(n2, x2.data, y2.data, z2.data);
	
	// extract parameters and arrays
	npy_intp nbins = PyArray_DIM(ac, 0) ;
	double *ac_data = (double *) PyArray_DATA(ac);
	npy_intp ac_stride = PyArray_STRIDE(ac, 0) ;
	
	size_t i1, i2, idx;
	double m = cos(maxangle);
	double d = nbins / maxangle;

#pragma omp parallel shared(x1, y1, z1, w1, x2, y2, z2, w2, n1, n2, d, nbins) private(i1, i2, idx)	
	{
		// setup temporary result array
		double *tmp = (double *) malloc(nbins * sizeof(double));
		int ib;
		for (ib = 0; ib < nbins; ib++) {
			tmp[ib] = 0.;
		}

		// compute correlation
#pragma omp for schedule(dynamic, 16) nowait
		for (i1 = 0; i1 < n1; i1++) {
			for (i2 = 0; i2 < n2; i2++) {
				float cosalpha = x1.data[i1] * x2.data[i2] + y1.data[i1] * y2.data[i2] + z1.data[i1] * z2.data[i2];
				if (cosalpha >= m) {
					idx = acos(cosalpha) * d;
					tmp[idx] += w1.data[i1] * w2.data[i2];
				}
			}
		}

		// add temporary result to total result
#pragma omp critical
		for (ib = 0; ib < nbins; ib++) {
			ac_data[ib] += tmp[ib];
		}

		// cleanup
		free(tmp);
	}

	free_fast_array(&x1);
	free_fast_array(&y1);
	free_fast_array(&z1);
	free_fast_array(&w1);
	free_fast_array(&x2);
	free_fast_array(&y2);
	free_fast_array(&z2);
	free_fast_array(&w2);

	Py_INCREF(Py_None);
	return Py_None;
}

static PyMethodDef _tpcMethods[] = {
	{ "tpac", tpac, METH_VARARGS },
	{ "tpcc", tpcc, METH_VARARGS },
	{ NULL,		NULL, 0 } 
};

void init_tpc() {
	Py_InitModule("_tpc", _tpcMethods);
	import_array();
}