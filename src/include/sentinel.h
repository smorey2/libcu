/*
sentinel.h - lite message bus framework for device to host functions
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

#pragma once
#ifndef _SENTINEL_H
#define _SENTINEL_H
#include <crtdefscu.h>
#include <host_defines.h>
#include <stdio.h>
#include <ext/pipeline.h>
#if _MSC_VER
#include <fcntl.h>
#include <io.h>
#endif
#ifdef __cplusplus
extern "C" {
#endif

#ifndef HAS_DEVICESENTINEL
#define HAS_DEVICESENTINEL 1
#endif
#ifndef HAS_HOSTSENTINEL
#define HAS_HOSTSENTINEL 1
#endif

#define SENTINEL_NAME "Sentinel" //"Global\\Sentinel"
#define SENTINEL_MAGIC (unsigned short)0xC811
#define SENTINEL_DEVICEMAPS 1
#define SENTINEL_MSGSIZE 4096
#define SENTINEL_MSGCOUNT 1

	struct sentinelMessage {
		bool Wait;
		unsigned short OP;
		int Size;
		char *(*Prepare)(void*, char*, char*, intptr_t);
		bool(*Postfix)(void*, intptr_t);
		__device__ sentinelMessage(bool wait, unsigned short op, int size = 0, char *(*prepare)(void*, char*, char*, intptr_t) = nullptr, bool(*postfix)(void*, intptr_t) = nullptr)
			: Wait(wait), OP(op), Size(size), Prepare(prepare), Postfix(postfix) { }
	public:
	};
#define SENTINELPREPARE(P) ((char *(*)(void*,char*,char*,intptr_t))&P)
#define SENTINELPOSTFIX(P) ((bool (*)(void*,intptr_t))&P)

	struct sentinelRedirect {
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
		__forceinline__ void doRedirect() {
			//: F0(_fileno(stdin)), F1(_fileno(stdout)), F2(_fileno(stderr)) {
			//FDTYPE p0, p1, p2;
			//CreatePipeline(0, nullptr, nullptr, &p0, &p1, &p2);
			//F0 = _get_osfhandle((int)p0);
			//F1 = _get_osfhandle((int)p1);
			//F2 = _get_osfhandle((int)p2);
			//printf("a1: %d, %d, %d\n", F0, F1, F2);
		}
		//https://stackoverflow.com/questions/5193579/how-make-file-from-handle-in-winapi
		__forceinline__ void toFiles(FILE **fs) {
			//printf("b: %d, %d, %d\n", F0, F1, F2);
			//fs[0] = _fdopen(_open_osfhandle(F0, _O_RDONLY), "r");
			//fs[1] = _fdopen(_open_osfhandle(F1, _O_WRONLY), "w");
			//fs[2] = _fdopen(_open_osfhandle(F2, _O_RDWR), "rw");
		}
#else
		//sentinelRedirect(int f0, int f1, int f2)
		//	: F0(f0), F1(f1), F2(f2) {
		//	printf("a0: %d, %d, %d\n", F0, F1, F2);
		//}
		sentinelRedirect()
			: F0(stdin), F1(stdout), F2(stderr) {
			printf("a: %d, %d, %d\n", F0, F1, F2);
		}
		__forceinline__ void toFiles(int **fs) {
			printf("b: %d, %d, %d\n", F0, F1, F2);
			fs[0] = fdopen(F0, _O_RDONLY);
			fs[1] = fdopen(F1, _O_WRONLY);
			fs[2] = fdopen(F2, _O_RDWR);
		}
#endif
	};

	struct sentinelClientMessage {
		sentinelMessage Base;
		sentinelRedirect Redir;
		sentinelClientMessage(bool wait, unsigned short op, int size = 0, char *(*prepare)(void*, char*, char*, intptr_t) = nullptr, bool(*postfix)(void*, intptr_t) = nullptr)
			: Base(wait, op, size, prepare, postfix), Redir() { }
	};

	typedef struct __align__(8) {
		unsigned short Magic;
		volatile long Control;
		int Length;
#ifndef _WIN64
		int Unknown;
#endif
		char Data[1];
		void Dump();
	} sentinelCommand;

	typedef struct __align__(8) {
		long GetId;
		volatile long SetId;
		intptr_t Offset;
		char Data[SENTINEL_MSGSIZE*SENTINEL_MSGCOUNT];
		void Dump();
	} sentinelMap;

	typedef struct sentinelExecutor {
		sentinelExecutor *Next;
		const char *Name;
		bool(*Executor)(void*, sentinelMessage*, int, char*(**)(void*, char*, char*, intptr_t));
		void *Tag;
	} sentinelExecutor;

	typedef struct sentinelContext {
		sentinelMap *DeviceMap[SENTINEL_DEVICEMAPS];
		sentinelMap *HostMap;
		sentinelExecutor *HostList;
		sentinelExecutor *DeviceList;
	} sentinelContext;

	//#if HAS_HOSTSENTINEL
	//	extern sentinelMap *_sentinelHostMap;
	//	extern intptr_t _sentinelHostMapOffset;
	//#endif
#if HAS_DEVICESENTINEL
	extern __constant__ const sentinelMap *_sentinelDeviceMap[SENTINEL_DEVICEMAPS];
#endif

	extern bool sentinelDefaultExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*, char*, char*, intptr_t));
	extern void sentinelServerInitialize(sentinelExecutor *executor = nullptr, char *mapHostName = (char *)SENTINEL_NAME, bool hostSentinel = true, bool deviceSentinel = true);
	extern void sentinelServerShutdown();
	extern sentinelRedirect sentinelRedirectClientMessage(sentinelMessage *data);
#if HAS_DEVICESENTINEL
	extern __device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength);
#endif
#if HAS_HOSTSENTINEL
	extern void sentinelClientInitialize(char *mapHostName = (char *)SENTINEL_NAME);
	extern void sentinelClientShutdown();
	extern void sentinelClientSend(sentinelMessage *msg, int msgLength);
#endif
	extern sentinelExecutor *sentinelFindExecutor(const char *name, bool forDevice = true);
	extern void sentinelRegisterExecutor(sentinelExecutor *exec, bool makeDefault = false, bool forDevice = true);
	extern void sentinelUnregisterExecutor(sentinelExecutor *exec, bool forDevice = true);

	// file-utils
	extern void sentinelRegisterFileUtils();

#ifdef  __cplusplus
}
#endif

#endif  /* _SENTINEL_H */