/*
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
*/

#define PY_SSIZE_T_CLEAN

#include <Python.h>

int _fib(int n) {
    return n < 2 ? n : _fib(n - 1) + _fib(n - 2);
}

static PyObject *hello(PyObject *self, PyObject *args) {
    const char *name;

    if (!PyArg_ParseTuple(args, "s", &name))
        return NULL;

    printf("Hello, %s!", name);

    return Py_None;
}

static PyObject *fib(PyObject *self, PyObject *args) {
    int n;

    if (!PyArg_ParseTuple(args, "i", &n))
        return NULL;

    return Py_BuildValue("i", _fib(n));
}

static PyMethodDef methods[] = {
    {"hello", hello, METH_VARARGS, "hello"},
    {"fib", fib, METH_VARARGS, "fib"},
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
