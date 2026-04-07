/*
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
*/

#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <arrayobject.h>

float _cdist(float x1, float x2) {
    return x1 - x2;
}

static PyObject *cdist(PyObject *self, PyObject *args) {
    float x1, x2;

    if (!PyArg_ParseTuple(args, "ff", &x1, &x2))
        return NULL;

    return Py_BuildValue("f", _cdist(x1, x2));
}

static PyMethodDef methods[] = {
    {"cdist", cdist, METH_VARARGS, "cdist"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_utils",
    .m_size = 0,
    .m_methods = methods
};

PyMODINIT_FUNC PyInit__utils_(void) {
    import_array();
    return PyModule_Create(&module);
}
