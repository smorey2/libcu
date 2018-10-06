#include "jim.h"
#include "jimautoconf.h"
#define HAVE_DLOPEN_COMPAT
#include <errno.h>

#if __CUDACC__
#elif defined(_WIN32) || defined(WIN32)
#ifndef STRICT
#define STRICT
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#if defined(HAVE_DLOPEN_COMPAT)
void *dlopen(const char *path, int mode) {
	JIM_NOTUSED(mode);
	return (void *)LoadLibraryA(path);
}

int dlclose(void *handle) {
	FreeLibrary((HMODULE)handle);
	return 0;
}

void *dlsym(void *handle, const char *symbol) {
	return GetProcAddress((HMODULE)handle, symbol);
}

char *dlerror() {
	static char msg[121];
	FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, NULL, GetLastError(), LANG_NEUTRAL, msg, sizeof(msg) - 1, NULL);
	return msg;
}
#endif

#ifdef _MSC_VER

//#include <sys/timeb.h>
///* POSIX gettimeofday() compatibility for WIN32 */
//int gettimeofday(struct timeval *tv, void *unused) //{
//	struct _timeb tb;
//
//	_ftime(&tb);
//	tv->tv_sec = tb.time;
//	tv->tv_usec = tb.millitm * 1000;
//
//	return 0;
//}

DIR *opendir(const char *name) {
	DIR *dir = 0;
	if (name && name[0]) {
		size_t base_length = strlen(name);
		const char *all = (strchr("/\\", name[base_length - 1]) ? "*" : "/*"); // search pattern must end with suitable wildcard
		if ((dir = (DIR *)Jim_Alloc(sizeof(*dir))) != 0 && (dir->name = (char *)Jim_Alloc((int)(base_length + strlen(all) + 1))) != 0) {
			strcat(strcpy(dir->name, name), all);
			if ((dir->handle = (long)_findfirst(dir->name, &dir->info)) != -1)
				dir->result.d_name = 0;
			else { // rollback
				Jim_Free(dir->name);
				Jim_Free(dir);
				dir = 0;
			}
		}
		else { // rollback
			Jim_Free(dir);
			dir = 0;
			errno = ENOMEM;
		}
	}
	else
		errno = EINVAL;
	return dir;
}

int closedir(DIR *dir) {
	int result = -1;
	if (dir) {
		if (dir->handle != -1)
			result = _findclose(dir->handle);
		Jim_Free(dir->name);
		Jim_Free(dir);
	}
	if (result == -1) // map all errors to EBADF
		errno = EBADF;
	return result;
}

struct dirent *readdir(DIR * dir) {
	struct dirent *result = 0;
	if (dir && dir->handle != -1) {
		if (!dir->result.d_name || _findnext(dir->handle, &dir->info) != -1) {
			result = &dir->result;
			result->d_name = dir->info.name;
		}
	}
	else
		errno = EBADF;
	return result;
}
#endif
#endif
