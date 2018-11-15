#include <jumbo.h>
#if __OS_WIN
#include <windows.h>
#elif __OS_UNIX
#include <stdlib.h>
#include <string.h>
#endif
#include <stdio.h>
#include <assert.h>
#include <cuda_runtimecu.h>

void jumboCommand::Dump() {
	register unsigned char *b = (unsigned char *)&Data;
	register int l = Length;
	printf("Cmd: %d[%d]'", 0, l); for (int i = 0; i < l; i++) printf("%02x", b[i] & 0xff); printf("'\n");
}

static jumboContext _ctx;

#if HAS_DEVICEJUMBO
static bool _jumboDevice = false;
static int *_deviceMap[JUMBO_DEVICEMAPS];
#endif
void jumboHostInitialize() {
#if HAS_DEVICEJUMBO
	// create device maps
	_jumboDevice = true;
	jumboMap *d_deviceMap[JUMBO_DEVICEMAPS];
	for (int i = 0; i < JUMBO_DEVICEMAPS; i++) {
		cudaErrorCheckF(cudaHostAlloc((void **)&_deviceMap[i], sizeof(jumboMap), cudaHostAllocPortable | cudaHostAllocMapped), goto initialize_error);
		d_deviceMap[i] = _ctx.DeviceMap[i] = (jumboMap *)_deviceMap[i];
		cudaErrorCheckF(cudaHostGetDevicePointer((void **)&d_deviceMap[i], _ctx.DeviceMap[i], 0), goto initialize_error);
#ifndef _WIN64
		_ctx.DeviceMap[i]->Offset = (intptr_t)((char *)_deviceMap[i] - (char *)d_deviceMap[i]);
		//printf("chk: %x %x [%x]\n", (char *)_deviceMap[i], (char *)d_deviceMap[i], _ctx.DeviceMap[i]->Offset);
#else
		_ctx.DeviceMap[i]->Offset = 0;
#endif
	}
	cudaErrorCheckF(cudaMemcpyToSymbol(_jumboDeviceMap, &d_deviceMap, sizeof(d_deviceMap)), goto initialize_error);
#endif
	return;
initialize_error:
	perror("jumboHostInitialize:Error");
	jumboHostInitialize();
	exit(1);
}

void jumboHostShutdown() {
	// close device maps
	for (int i = 0; i < JUMBO_DEVICEMAPS; i++) {
		if (_deviceMap[i]) { cudaErrorCheckA(cudaFreeHost(_deviceMap[i])); _deviceMap[i] = nullptr; }
	}
}
