// src/py_silero_vad.c

#define Py_LIMITED_API 0x03090000
#include <Python.h>
#include <stdint.h>

#include "whisper_vad.hpp"

#define SILERO_VAD_CAPSULE_NAME "silero_vad.ctx"

// ---- Capsule helpers ----

static void silero_vad_capsule_destructor(PyObject *capsule) {
    whisper_vad_context *vctx = (whisper_vad_context *)PyCapsule_GetPointer(
        capsule, SILERO_VAD_CAPSULE_NAME);
    if (vctx) {
        whisper_vad_free(vctx);
    }
}

static whisper_vad_context *get_ctx_from_capsule(PyObject *capsule) {
    return (whisper_vad_context *)PyCapsule_GetPointer(capsule,
                                                       SILERO_VAD_CAPSULE_NAME);
}

// ---- load_model(path: str) -> capsule ----

static PyObject *py_silero_vad_load_model(PyObject *self, PyObject *args) {
    (void)self;
    const char *path = NULL;

    if (!PyArg_ParseTuple(args, "s", &path)) {
        return NULL;
    }

    whisper_vad_context_params vctx_params =
        whisper_vad_default_context_params();

    whisper_vad_context *vctx =
        whisper_vad_init_from_file_with_params(path, vctx_params);

    if (!vctx) {
        PyErr_Format(PyExc_OSError, "Failed to load Silero VAD model from '%s'",
                     path);
        return NULL;
    }

    PyObject *capsule = PyCapsule_New((void *)vctx, SILERO_VAD_CAPSULE_NAME,
                                      silero_vad_capsule_destructor);
    if (!capsule) {
        whisper_vad_free(vctx);
        return NULL;
    }

    return capsule;
}

// ---- reset(ctx_capsule) -> None ----

static PyObject *py_silero_vad_reset(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;

    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return NULL;
    }

    whisper_vad_context *vctx = get_ctx_from_capsule(capsule);
    if (!vctx) {
        PyErr_SetString(PyExc_TypeError, "Invalid Silero VAD context capsule");
        return NULL;
    }

    whisper_vad_reset_state(vctx);

    Py_RETURN_NONE;
}

static int py_read_float_sequence(PyObject *obj, float **out_data,
                                  Py_ssize_t *out_len) {
    if (!PySequence_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "expected a sequence of floats");
        return -1;
    }

    Py_ssize_t n = PySequence_Size(obj);
    if (n < 0) {
        // PySequence_Size sets an exception on error
        return -1;
    }

    float *buf = (float *)PyMem_Malloc((size_t)n * sizeof(float));
    if (!buf) {
        PyErr_NoMemory();
        return -1;
    }

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = PySequence_GetItem(obj, i); // new ref
        if (!item) {
            PyMem_Free(buf);
            return -1; // error already set
        }

        double v = PyFloat_AsDouble(item); // accepts ints too
        Py_DECREF(item);

        if (PyErr_Occurred()) {
            PyMem_Free(buf);
            return -1;
        }

        buf[i] = (float)v;
    }

    *out_data = buf;
    *out_len = n;
    return 0;
}

static PyObject *py_silero_vad_process_chunk(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    PyObject *samples_obj = NULL;

    if (!PyArg_ParseTuple(args, "OO", &capsule, &samples_obj)) {
        return NULL;
    }

    whisper_vad_context *vctx = get_ctx_from_capsule(capsule);
    if (!vctx) {
        PyErr_SetString(PyExc_TypeError, "Invalid Silero VAD context capsule");
        return NULL;
    }

    float *samples = NULL;
    Py_ssize_t n_samples = 0;

    if (py_read_float_sequence(samples_obj, &samples, &n_samples) < 0) {
        // error already set
        return NULL;
    }

    PyObject *result = NULL;

    if (n_samples == vctx->n_window) {
        float prob = 0.0f;
        bool rc =
            whisper_vad_process_chunk(vctx, samples, (int)n_samples, &prob);

        if (rc) {
            result = PyFloat_FromDouble(prob);
        }
    }

    PyMem_Free(samples);

    if (!result) {

        PyErr_SetString(PyExc_ValueError,
                        "Invalid number of samples (expected 512)");
        return NULL;
    }

    return result;
}

// ---- Module definition ----

static PyMethodDef silero_vad_methods[] = {
    {"load_model", (PyCFunction)py_silero_vad_load_model, METH_VARARGS,
     "Load a Silero VAD ggml model from a file and return a context capsule."},
    {"reset", (PyCFunction)py_silero_vad_reset, METH_VARARGS,
     "Reset recurrent state of a Silero VAD context."},
    {"process_chunk", (PyCFunction)py_silero_vad_process_chunk, METH_VARARGS,
     "Process 512 float samples and return speech probability (0.0-1.0)."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef silero_vad_module = {
    PyModuleDef_HEAD_INIT,
    /*.m_name =*/"_silero_vad",
    /*.m_doc =*/"Silero VAD ggml backend (abi3, limited API).",
    /*.m_size =*/-1,
    /*.m_methods =*/silero_vad_methods,
};

PyMODINIT_FUNC PyInit_silero_vad(void) {
    return PyModule_Create(&silero_vad_module);
}
