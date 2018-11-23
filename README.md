# libcu
The CU standard library or libcu is the standard library for the C programming language implemented in CUDA, as specified in the ANSI C standard.

## Quick Start

Sentinel needs to be started, and shutdown for any host to device access, but is not required. An error will throw telling you to start sentinel if access is attempted.

A simple project startup looks like the following:
```
#include <cuda_runtimecu.h>
#include <sentinel.h>
...
// choose which GPU to run on and extend stack size (optional)
cudaErrorCheck(cudaSetDevice(gpuGetMaxGflopsDevice()));
cudaErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 5));
// main body
sentinelServerInitialize();
...
code
...
sentinelServerShutdown();
```

## Documents

This project follows the stardard libc interface.
* Learn by Reference documentation can be found in [docs](https://github.com/libcu/libcu/tree/master/docs).
* Learn by Tests can be found in [libcu.tests](https://github.com/libcu/libcu/tree/master/src/libcu.tests).

## Contributing

If you would like to contribute to libcu to help the project along, consider these options:
* Improve our [documentation](https://github.com/libcu/libcu/tree/master/docs). Documentation often gets over looked, but can be a large contributer to the success of a project.
* Submit a [bug report](https://github.com/libcu/libcu/issues) (for an excellent guide on submitting good bug reports, read [Painless Bug Tracking](https://www.joelonsoftware.com/2000/11/08/painless-bug-tracking/)).
* Submit a [feature request](https://github.com/libcu/libcu/issues).
* Help verify submitted fixes for bugs.
* Help answer questions in the discussions list.
* Submit or update a unit test to help provide code coverage.
* Tell others about the project.
* Tell the developers how your using the product, or how much you appreciate the product!

## General Guidelines
Please only contribute code which you wrote or have the rights to contribute. This project is licensed under the MIT license.
