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
#include <driver_types.h>
//#include <host_defines.h>
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
#define SENTINEL_MSGSIZE 5120
#define SENTINEL_MSGCOUNT 1
#define SENTINEL_CHUNK 4096

	typedef struct sentinelInPtr {
		void *field;
		int size;
		void *unknown;
	} sentinelInPtr;

	typedef struct sentinelOutPtr {
		void *field;
		void *buf;
		int size;
		void *sizeField;
		void *unknown;
	} sentinelOutPtr;

#define SENTINELFLOW_NONE 0
#define SENTINELFLOW_WAIT 1

	typedef struct sentinelMessage {
		unsigned short op;
		unsigned char flow;
		//unsigned char unknown;
		int size;
		char *(*prepare)(void*, char*, char*, intptr_t);
		bool(*postfix)(void*, intptr_t);
		__device__ sentinelMessage(unsigned short op, unsigned char flow = SENTINELFLOW_WAIT, int size = 0, char *(*prepare)(void*, char*, char*, intptr_t) = nullptr, bool(*postfix)(void*, intptr_t) = nullptr)
			: op(op), flow(flow), size(size), prepare(prepare), postfix(postfix) { }
	} sentinelMessage;
#define SENTINELPREPARE(P) ((char *(*)(void*,char*,char*,intptr_t))&P)
#define SENTINELPOSTFIX(P) ((bool (*)(void*,intptr_t))&P)

	typedef struct sentinelClientMessage {
		sentinelMessage base;
		pipelineRedir redir;
		sentinelClientMessage(pipelineRedir redir, unsigned short op, unsigned char flow = SENTINELFLOW_WAIT, int size = 0, char *(*prepare)(void*, char*, char*, intptr_t) = nullptr, bool(*postfix)(void*, intptr_t) = nullptr)
			: base(op, flow, size, prepare, postfix), redir(redir) { }
	} sentinelClientMessage;

	typedef struct __align__(8) {
		unsigned short magic;
		volatile long control;
		int unknown;
		int length;
		char data[1];
		void dump();
	} sentinelCommand;

	typedef struct __align__(8) {
		long getId;
		volatile long setId;
		intptr_t offset;
		char data[SENTINEL_MSGSIZE*SENTINEL_MSGCOUNT];
		void dump();
	} sentinelMap;

	typedef struct sentinelExecutor {
		sentinelExecutor *next;
		const char *name;
		bool(*executor)(void*, sentinelMessage*, int, char*(**)(void*, char*, char*, intptr_t));
		void *tag;
	} sentinelExecutor;

	typedef struct sentinelContext {
		sentinelMap *deviceMap[SENTINEL_DEVICEMAPS];
		sentinelMap *hostMap;
		sentinelExecutor *hostList;
		sentinelExecutor *deviceList;
	} sentinelContext;

//#if HAS_HOSTSENTINEL // not-required
//	extern sentinelMap *_sentinelHostMap;
//	extern intptr_t _sentinelHostMapOffset;
//#endif
#if HAS_DEVICESENTINEL
	extern __constant__ const sentinelMap *_sentinelDeviceMap[SENTINEL_DEVICEMAPS];
#endif

	extern bool sentinelDefaultHostExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*, char*, char*, intptr_t));
	extern bool sentinelDefaultDeviceExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*, char*, char*, intptr_t));
	extern void sentinelServerInitialize(sentinelExecutor *deviceExecutor = nullptr, char *mapHostName = (char *)SENTINEL_NAME, bool hostSentinel = true, bool deviceSentinel = true);
	extern void sentinelServerShutdown();
#if HAS_DEVICESENTINEL
	extern __device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength, sentinelInPtr *ptrsIn = nullptr, sentinelOutPtr *ptrsOut = nullptr);
#endif
#if HAS_HOSTSENTINEL
	extern void sentinelClientInitialize(char *mapHostName = (char *)SENTINEL_NAME);
	extern void sentinelClientShutdown();
	extern void sentinelClientRedir(pipelineRedir *redir);
	extern void sentinelClientSend(sentinelMessage *msg, int msgLength, sentinelInPtr *ptrsIn = nullptr, sentinelOutPtr *ptrsOut = nullptr);
#endif
	extern sentinelExecutor *sentinelFindExecutor(const char *name, bool forDevice = true);
	extern void sentinelRegisterExecutor(sentinelExecutor *exec, bool makeDefault = false, bool forDevice = true);
	extern void sentinelUnregisterExecutor(sentinelExecutor *exec, bool forDevice = true);

	// file-utils
	extern void sentinelRegisterFileUtils();

#define SENTINELCONTROL_NORMAL 0
#define SENTINELCONTROL_DEVICE 1
#define SENTINELCONTROL_DEVICERDY 2
#define SENTINELCONTROL_DEVICEWAIT 3
#define SENTINELCONTROL_DEVICEDONE 4
#define SENTINELCONTROL_HOST 5
#define SENTINELCONTROL_HOSTRDY 6
#define SENTINELCONTROL_HOSTWAIT 7

#ifdef  __cplusplus
}
#endif

#endif  /* _SENTINEL_H */