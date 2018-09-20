/*
pipeline.h - xxx
The MIT License

Copyright (c) 2016 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef _EXT_PIPELINE_H
#define _EXT_PIPELINE_H
#ifdef  __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
#ifndef STRICT
#define STRICT
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
	typedef HANDLE FDTYPE;
	typedef HANDLE PIDTYPE;
#define __BAD_FD INVALID_HANDLE_VALUE
#define __BAD_PID INVALID_HANDLE_VALUE
#else
	typedef int FDTYPE;
	typedef int PIDTYPE;
#define __BAD_FD -1
#define __BAD_PID -1
#endif


	struct pipelineRedirect {
		int F0; int F1; int F2;
#if _MSC_VER
		//sentinelRedirect(int f0, int f1, int f2)
		//	: F0(_get_osfhandle(f0)), F1(_get_osfhandle(f1)), F2(_get_osfhandle(f2)) {
		//	printf("a0: %d, %d, %d\n", F0, F1, F2);
		//}
		//__forceinline__ sentinelRedirect() {
		//	//: F0(_fileno(stdin)), F1(_fileno(stdout)), F2(_fileno(stderr)) {
		//	FDTYPE p0, p1, p2;
		//	CreatePipeline(0, nullptr, nullptr, &p0, &p1, &p2);
		//	F0 = _get_osfhandle((int)p0);
		//	F1 = _get_osfhandle((int)p1);
		//	F2 = _get_osfhandle((int)p2);
		//	printf("a1: %d, %d, %d\n", F0, F1, F2);
		//}
		//__forceinline__ void doRedirect() {
			//: F0(_fileno(stdin)), F1(_fileno(stdout)), F2(_fileno(stderr)) {
			//FDTYPE p0, p1, p2;
			//CreatePipeline(0, nullptr, nullptr, &p0, &p1, &p2);
			//F0 = _get_osfhandle((int)p0);
			//F1 = _get_osfhandle((int)p1);
			//F2 = _get_osfhandle((int)p2);
			//printf("a1: %d, %d, %d\n", F0, F1, F2);
		//}
		//https://stackoverflow.com/questions/5193579/how-make-file-from-handle-in-winapi
		//__forceinline__ void toFiles(FILE **fs) {
			//printf("b: %d, %d, %d\n", F0, F1, F2);
			//fs[0] = _fdopen(_open_osfhandle(F0, _O_RDONLY), "r");
			//fs[1] = _fdopen(_open_osfhandle(F1, _O_WRONLY), "w");
			//fs[2] = _fdopen(_open_osfhandle(F2, _O_RDWR), "rw");
		//}
#else
		//sentinelRedirect(int f0, int f1, int f2)
		//	: F0(f0), F1(f1), F2(f2) {
		//	printf("a0: %d, %d, %d\n", F0, F1, F2);
		//}
		// pipelineRedirect()
		// 	: F0(stdin), F1(stdout), F2(stderr) {
		// 	printf("a: %d, %d, %d\n", F0, F1, F2);
		// }
		// __forceinline__ void toFiles(int **fs) {
		// 	printf("b: %d, %d, %d\n", F0, F1, F2);
		// 	fs[0] = fdopen(F0, _O_RDONLY);
		// 	fs[1] = fdopen(F1, _O_WRONLY);
		// 	fs[2] = fdopen(F2, _O_RDWR);
		// }
#endif
	};

	/* Cleanup Children */
	extern int CleanupChildren(int numPids, PIDTYPE *pids, int child_siginfo);
	/* Create Pipeline */
	extern int CreatePipeline(int argc, char **argv, PIDTYPE **pidsPtr, FDTYPE *inPipePtr, FDTYPE *outPipePtr, FDTYPE *errFilePtr);

#ifdef  __cplusplus
}
#endif
#endif  /* _EXT_PIPELINE_H */