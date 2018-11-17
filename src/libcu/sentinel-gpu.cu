#include <crtdefscu.h>
#include <stdiocu.h>
#include <stdlibcu.h>
#include <sentinel.h>

__BEGIN_DECLS;

#if HAS_DEVICESENTINEL

__device__ volatile unsigned int _sentinelMapId;
__constant__ const sentinelMap *_sentinelDeviceMap[SENTINEL_DEVICEMAPS];
__device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength) {
	unsigned int s_;
	const sentinelMap *map = _sentinelDeviceMap[_sentinelMapId++ % SENTINEL_DEVICEMAPS];
	if (!map)
		panic("sentinel: device map not defined. did you start sentinel?\n");

	// ATTACH
	long id = atomicAdd((int *)&map->SetId, SENTINEL_MSGSIZE);
	sentinelCommand *cmd = (sentinelCommand *)&map->Data[id % sizeof(map->Data)]; //cmd->Data = (char *)cmd + ROUND8_(sizeof(sentinelCommand));
	if (cmd->Magic != SENTINEL_MAGIC)
		panic("Bad Sentinel Magic");
	volatile long *control = (volatile long *)&cmd->Control;
	/* spin-lock */ do { s_ = *control; /*printf("%d ", s_);*/ __syncthreads(); } while (s_ != 0); __syncthreads();
	//*control = 1; // device in-progress

	// PREPARE
	cmd->Length = msgLength;
	if (msg->Prepare && !msg->Prepare(msg, cmd->Data, cmd->Data + ROUND8_(msgLength) + msg->Size, map->Offset))
		panic("msg too long");
	// FLOW-OUT
	if (msg->Flow & FLOW_JUMBOOUT) {
		while (true) {
			/* spin-lock */ do { s_ = *control; /*printf("%d ", s_);*/ __syncthreads(); } while (s_ != 10); __syncthreads();
		}
	}
	memcpy(cmd->Data, msg, msgLength);
	//printf("Msg: %d[%d]'", msg->OP, msgLength); for (int i = 0; i < msgLength; i++) printf("%02x", ((char *)msg)[i] & 0xff); printf("'\n");
	*control = 2; // client signal that data is ready to process

	// FLOW-IN
	if (msg->Flow & FLOW_JUMBOIN) {
		while (true) {
			/* spin-lock */ do { s_ = *control; /*printf("%d ", s_);*/ __syncthreads(); } while (s_ != 10); __syncthreads();
		}
	}

	// FLOW-WAIT
	if (msg->Flow & FLOW_WAIT) {
		/* spin-lock */ do { s_ = *control; /*printf("%d ", s_);*/ __syncthreads(); } while (s_ != 4); __syncthreads();
		memcpy(msg, cmd->Data, msgLength);
		if (msg->Postfix && !msg->Postfix(msg, map->Offset))
			panic("postfix error");
	}
	*control = 0; // normal state
}

#endif

__END_DECLS;
