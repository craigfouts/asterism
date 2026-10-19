/*
Authors: Craig Fouts
License: Apache 2.0 license
*/

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

// float _sum(float *)

float _test(float x1, float x2) {
    return x1 - x2;
}

static PyObject *test(PyObject *self, PyObject *args) {
    float x1, x2;

    if (!PyArg_ParseTuple(args, "ff", &x1, &x2))
        return NULL;

    return Py_BuildValue("f", _test(x1, x2));

    // PyObject *arg1 = NULL;
    // PyObject *arr1 = NULL;
    // int nd;

    // if (!PyArg_ParseTuple(args, "O", &arg1))
    //     return NULL;

    // arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
    // nd = PyArray_NDIM(arr1);
    // npy_intp *sp = PyArray_SHAPE(arr1);
}

static PyMethodDef methods[] = {
    {"test", test, METH_VARARGS, "test"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_utils",
    .m_size = 0,
    .m_methods = methods
};

PyMODINIT_FUNC PyInit__utils_(void) {
    return PyModule_Create(&module);
}
