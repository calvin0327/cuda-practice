#pragma once

#include <Python.h>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

// A version of the TORCH_LIBRARY macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

// A version of the TORCH_LIBRARY_IMPL macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
#define TORCH_LIBRARY_IMPL_EXPAND(NAME, DEVICE, MODULE) \
  TORCH_LIBRARY_IMPL(NAME, DEVICE, MODULE)

// extern "C" {
// /* Creates a dummy empty _C module that can be imported from Python.
//    The import from Python will load the .so consisting of this file
//    in this extension, so that the TORCH_LIBRARY static initializers
//    below are run. */
// PyObject* PyInit__C(void) {
//   static struct PyModuleDef module_def = {
//       PyModuleDef_HEAD_INIT,
//       "_C", /* name of module */
//       NULL, /* module documentation, may be NULL */
//       -1,   /* size of per-interpreter state of the module,
//               or -1 if the module keeps state in global variables. */
//       NULL, /* methods */
//   };
//   return PyModule_Create(&module_def);
// }
// }

// REGISTER_EXTENSION allows the shared library to be loaded and initialized
// via python's import statement.
#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }
