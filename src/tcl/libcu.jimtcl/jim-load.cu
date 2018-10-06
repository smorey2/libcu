//#include <string.h>
#include "jimautoconf.h"
#include "jim.h"

// -----------------------------------------------------------------------------
// Dynamic libraries support (WIN32 not supported)
// -----------------------------------------------------------------------------
#pragma region Dynamic libraries support (WIN32 not supported)

#if !defined(__CUDACC__) && (defined(HAVE_DLOPEN) || defined(HAVE_DLOPEN_COMPAT))

#ifdef HAVE_DLFCN_H
#include <dlfcn.h>
#endif

#ifndef RTLD_NOW
#define RTLD_NOW 0
#endif
#ifndef RTLD_LOCAL
#define RTLD_LOCAL 0
#endif

static void JimFreeLoadHandles(Jim_Interp *interp, void *data);

// Note that Jim_LoadLibrary() requires a path to an existing file.
// If it is necessary to search JIM_LIBPATH, use Jim_PackageRequire() instead.
int Jim_LoadLibrary(Jim_Interp *interp, const char *pathName) {
	void *handle = dlopen(pathName, RTLD_NOW | RTLD_LOCAL);
	if (handle == NULL)
		Jim_SetResultFormatted(interp, "error loading extension \"%s\": %s", pathName, dlerror());
	else {
		// We use a unique init symbol depending on the extension name. This is done for compatibility between static and dynamic extensions. For extension readline.so, the init symbol is "Jim_readlineInit"
		typedef int jim_module_init_func_type(Jim_Interp *);

		const char *pt = strrchr(pathName, '/');
		const char *pkgname = (pt ? pt + 1 : pathName);
		pt = strchr(pkgname, '.');
		int pkgnamelen = (int)(pt ? pt - pkgname : strlen(pkgname));
		char initsym[40];
		snprintf(initsym, sizeof(initsym), "Jim_%.*sInit", pkgnamelen, pkgname);
		jim_module_init_func_type *onload;
		if ((onload = (jim_module_init_func_type *)dlsym(handle, initsym)) == NULL)
			Jim_SetResultFormatted(interp, "No %s symbol found in extension %s", initsym, pathName);
		else if (onload(interp) != JIM_ERROR) {
			// Add this handle to the stack of handles to be freed
			Jim_Stack *loadHandles = (Jim_Stack *)Jim_GetAssocData(interp, "load::handles");
			if (loadHandles == NULL) {
				loadHandles = (Jim_Stack *)Jim_Alloc(sizeof(*loadHandles));
				Jim_InitStack(loadHandles);
				Jim_SetAssocData(interp, "load::handles", JimFreeLoadHandles, loadHandles);
			}
			Jim_StackPush(loadHandles, handle);
			Jim_ResetResult(interp);
			return JIM_OK;
		}
	}
	if (handle)
		dlclose(handle);
	return JIM_ERROR;
}

static void JimFreeOneLoadHandle(void *handle) {
	dlclose(handle);
}

static void JimFreeLoadHandles(Jim_Interp *interp, void *data) {
	Jim_Stack *handles = (Jim_Stack *)data;
	if (handles) {
		Jim_FreeStackElements(handles, JimFreeOneLoadHandle);
		Jim_FreeStack(handles);
		Jim_Free(handles);
	}
}

#else
__device__ int Jim_LoadLibrary(Jim_Interp *interp, const char *pathName) {
	JIM_NOTUSED(interp);
	JIM_NOTUSED(pathName);
	Jim_SetResultString(interp, "the Jim binary has no support for [load]", -1);
	return JIM_ERROR;
}

__device__ void Jim_FreeLoadHandles(Jim_Interp *interp) {
}
#endif

// [load]
static __device__ int Jim_LoadCoreCommand(ClientData dummy, Jim_Interp *interp, int argc, Jim_Obj *const *argv) {
	if (argc < 2) {
		Jim_WrongNumArgs(interp, 1, argv, "libraryFile");
		return JIM_ERROR;
	}
	return Jim_LoadLibrary(interp, Jim_String(argv[1]));
}

__device__ int Jim_loadInit(Jim_Interp *interp) {
	Jim_CreateCommand(interp, "load", Jim_LoadCoreCommand, NULL, NULL);
	return JIM_OK;
}

#pragma endregion