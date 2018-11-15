/*
jumbo.h - memory transfer framework for device to host functions
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
#ifndef _JUMBO_H
#define _JUMBO_H
#include <crtdefscu.h>
#include <stdio.h>
#if _MSC_VER
#include <fcntl.h>
#include <io.h>
#endif
#ifdef __cplusplus
extern "C" {
#endif

#ifndef HAS_DEVICEJUMBO
#define HAS_DEVICEJUMBO 1
#endif

#define JUMBO_MAGIC (unsigned short)0xC812
#define JUMBO_DEVICEMAPS 1
#define JUMBO_MSGSIZE 4096
#define JUMBO_MSGCOUNT 1

	typedef struct __align__(8) {
		unsigned short Magic;
		volatile long Control;
		int Length;
#ifndef _WIN64
		int Unknown;
#endif
		char Data[1];
		void Dump();
	} jumboCommand;

	typedef struct __align__(8) {
		long GetId;
		volatile long SetId;
		intptr_t Offset;
		char Data[JUMBO_MSGSIZE*JUMBO_MSGCOUNT];
		void Dump();
	} jumboMap;

	typedef struct jumboContext {
		jumboMap *DeviceMap[JUMBO_DEVICEMAPS];
		jumboMap *HostMap;
	} jumboContext;

#if HAS_DEVICEJUMBO
	extern __constant__ const jumboMap *_jumboDeviceMap[JUMBO_DEVICEMAPS];
#endif

	extern void jumboHostInitialize();
	extern void jumboHostShutdown();
#if HAS_DEVICEJUMBO
	extern __device__ void jumboDeviceSend(void *msg, int msgLength);
#endif

#ifdef  __cplusplus
}
#endif

#endif  /* JUMBO_H */