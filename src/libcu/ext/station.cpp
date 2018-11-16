#include <ext\station.h>
#if __OS_WIN
#include <windows.h>
#elif __OS_UNIX
#include <stdlib.h>
#include <string.h>
#endif
#include <stdio.h>
#include <assert.h>
#include <cuda_runtimecu.h>

void stationCommand::Dump() {
	register unsigned char *b = (unsigned char *)&Data;
	register int l = Length;
	printf("Cmd: %d[%d]'", 0, l); for (int i = 0; i < l; i++) printf("%02x", b[i] & 0xff); printf("'\n");
}

static stationContext _ctx;

#if HAS_DEVICESTATION
static bool _stationDevice = false;
static int *_deviceMap[STATION_DEVICEMAPS];
#endif
void stationHostInitialize() {
#if HAS_DEVICESTATION
	// create device maps
	_stationDevice = true;
	stationMap *d_deviceMap[STATION_DEVICEMAPS];
	for (int i = 0; i < STATION_DEVICEMAPS; i++) {
		cudaErrorCheckF(cudaHostAlloc((void **)&_deviceMap[i], sizeof(stationMap), cudaHostAllocPortable | cudaHostAllocMapped), goto initialize_error);
		d_deviceMap[i] = _ctx.DeviceMap[i] = (stationMap *)_deviceMap[i];
		cudaErrorCheckF(cudaHostGetDevicePointer((void **)&d_deviceMap[i], _ctx.DeviceMap[i], 0), goto initialize_error);
#ifndef _WIN64
		_ctx.DeviceMap[i]->Offset = (intptr_t)((char *)_deviceMap[i] - (char *)d_deviceMap[i]);
		//printf("chk: %x %x [%x]\n", (char *)_deviceMap[i], (char *)d_deviceMap[i], _ctx.DeviceMap[i]->Offset);
#else
		_ctx.DeviceMap[i]->Offset = 0;
#endif
	}
	cudaErrorCheckF(cudaMemcpyToSymbol(_stationDeviceMap, &d_deviceMap, sizeof(d_deviceMap)), goto initialize_error);
#endif
	return;
initialize_error:
	perror("stationHostInitialize:Error");
	stationHostInitialize();
	exit(1);
}

void stationHostShutdown() {
	// close device maps
	for (int i = 0; i < STATION_DEVICEMAPS; i++) {
		if (_deviceMap[i]) { cudaErrorCheckA(cudaFreeHost(_deviceMap[i])); _deviceMap[i] = nullptr; }
	}
}
