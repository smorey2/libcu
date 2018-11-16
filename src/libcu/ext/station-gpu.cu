#include <crtdefscu.h>
#include <stdiocu.h>
#include <stdlibcu.h>
#include <station.h>

__BEGIN_DECLS;

#if HAS_DEVICESTATION

__device__ volatile unsigned int _stationMapId;
__constant__ const stationMap *_stationDeviceMap[JUMBO_DEVICEMAPS];
__device__ void stationDeviceSend(void *msg, int msgLength) {
	const stationMap *map = _stationDeviceMap[_stationMapId++ % JUMBO_DEVICEMAPS];
	if (!map)
		panic("station: device map not defined. did you start station?\n");
	long id = atomicAdd((int *)&map->SetId, JUMBO_MSGSIZE);
	stationCommand *cmd = (stationCommand *)&map->Data[id % sizeof(map->Data)];
	volatile long *control = (volatile long *)&cmd->Control;
	//cmd->Data = (char *)cmd + ROUND8_(sizeof(stationCommand));
	cmd->Magic = STATION_MAGIC;
	cmd->Length = msgLength;
	//if (msg->Prepare && !msg->Prepare(msg, cmd->Data, cmd->Data + ROUND8_(msgLength) + msg->Size, map->Offset))
	//	panic("msg too long");
	//memcpy(cmd->Data, msg, msgLength);
	////printf("Msg: %d[%d]'", msg->OP, msgLength); for (int i = 0; i < msgLength; i++) printf("%02x", ((char *)msg)[i] & 0xff); printf("'\n");

	//*control = 2;
	//if (msg->Wait) {
	//	unsigned int s_; do { s_ = *control; /*printf("%d ", s_);*/ __syncthreads(); } while (s_ != 4); __syncthreads();
	//	memcpy(msg, cmd->Data, msgLength);
	//	*control = 0;
	//	if (msg->Postfix && !msg->Postfix(msg, map->Offset))
	//		panic("postfix error");
	//}
}

#endif

__END_DECLS;
