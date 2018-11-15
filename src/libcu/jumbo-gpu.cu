#include <crtdefscu.h>
#include <stdiocu.h>
#include <stdlibcu.h>
#include <jumbo.h>

__BEGIN_DECLS;

#if HAS_DEVICEJUMBO

__device__ volatile unsigned int _jumboMapId;
__constant__ const jumboMap *_jumboDeviceMap[JUMBO_DEVICEMAPS];
__device__ void jumboDeviceSend(void *msg, int msgLength) {
	const jumboMap *map = _jumboDeviceMap[_jumboMapId++ % JUMBO_DEVICEMAPS];
	if (!map)
		panic("jumbo: device map not defined. did you start jumbo?\n");
	long id = atomicAdd((int *)&map->SetId, JUMBO_MSGSIZE);
	jumboCommand *cmd = (jumboCommand *)&map->Data[id % sizeof(map->Data)];
	volatile long *control = (volatile long *)&cmd->Control;
	//cmd->Data = (char *)cmd + ROUND8_(sizeof(sentinelCommand));
	cmd->Magic = JUMBO_MAGIC;
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
