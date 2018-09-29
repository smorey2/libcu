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

#include <stdio.h>
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

	struct pipelineRedir {
		FILE *in;
		FILE *out;
		FILE *err;
		FDTYPE Input;
		FDTYPE Output;
		FDTYPE Error;
		/* Ack Pipeline */
		void Open();
		void Close();
		/* Read Pipeline */
		void Read();
	};

	/* Cleanup Children */
	extern int CleanupChildren(int numPids, PIDTYPE *pids, int child_siginfo);
	/* Create Pipeline */
	extern int CreatePipeline(int argc, char **argv, PIDTYPE **pidsPtr, FDTYPE *inPipePtr, FDTYPE *outPipePtr, FDTYPE *errFilePtr, FDTYPE process, pipelineRedir *redirs);

#ifdef  __cplusplus
}
#endif
#endif  /* _EXT_PIPELINE_H */