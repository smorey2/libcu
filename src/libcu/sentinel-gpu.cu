#include <crtdefscu.h>
#include <stringcu.h>
#include <sentinel.h>
#include <ext/mutex.h>

__BEGIN_DECLS;

#if HAS_DEVICESENTINEL

static __device__ void executeTrans(char id, sentinelCommand *cmd, int size, sentinelInPtr *listIn, sentinelOutPtr *listOut, intptr_t offset, char *&trans);

static __device__ char *preparePtrs(sentinelInPtr *ptrsIn, sentinelOutPtr *ptrsOut, sentinelCommand *cmd, char *data, char *dataEnd, intptr_t offset, sentinelOutPtr *&listOut_, char *&trans) {
	sentinelInPtr *i; sentinelOutPtr *o; char **field; char *ptr = data, *next;

	// PREPARE & TRANSFER
	int transSize = 0;
	sentinelInPtr *listIn = nullptr;
	if (ptrsIn)
		for (i = ptrsIn, field = (char **)i->field; field; i++, field = (char **)i->field) {
			if (!*field) { i->unknown = nullptr; continue; }
			int size = i->size != -1 ? i->size : (i->size = (int)strlen(*field) + 1);
			next = ptr + size;
			if (!size) i->unknown = nullptr;
			else if (next <= dataEnd) { i->unknown = ptr; ptr = next; }
			else { i->unknown = listIn; listIn = i; transSize += size; }
		}
	sentinelOutPtr *listOut = nullptr;
	if (ptrsOut) {
		if (ptrsOut[0].field != (char *)-1) ptr = data;
		else { ptrsOut[0].unknown = nullptr; ptrsOut++; } // { -1 } = append
		for (o = ptrsOut, field = (char **)o->field; field; o++, field = (char **)o->field) {
			int size = o->size != -1 ? o->size : (o->size = (int)(dataEnd - ptr));
			next = ptr + size;
			if (!size) o->unknown = nullptr;
			else if (next <= dataEnd) { o->unknown = ptr; ptr = next; }
			else { o->unknown = listOut; listOut = o; transSize += size; }
		}
	}
	listOut_ = listOut;

	// TRANSFER & PACK
	if (transSize) executeTrans(0, cmd, transSize, listIn, listOut, offset, trans); // size & transfer-in
	if (ptrsIn)
		for (i = ptrsIn, field = (char **)i->field; field; i++, field = (char **)i->field) {
			if (!*field || !(ptr = (char *)i->unknown)) continue;
			memcpy(ptr, *field, i->size); // transfer-in
			*field = ptr + offset;
		}
	if (ptrsOut)
		for (o = ptrsOut, field = (char **)o->field; field; o++, field = (char **)o->field) {
			if (!(ptr = (char *)o->unknown)) continue;
			*field = ptr + offset;
		}
	return data;
}

static __device__ bool postfixPtrs(sentinelOutPtr *ptrsOut, sentinelCommand *cmd, intptr_t offset, sentinelOutPtr *listOut, char *&trans) {
	sentinelOutPtr *o; char **field, **buf; char *ptr;
	// UNPACK & TRANSFER
	if (ptrsOut) {
		if (ptrsOut[0].field == (char *)-1) ptrsOut++; // { -1 } = append
		for (o = ptrsOut, field = (char **)o->field; field; o++, field = (char **)o->field) {
			if (!*field || !(buf = (char **)o->buf)) continue;
			int *sizeField = (int *)o->sizeField;
			int size = !sizeField ? o->size : *sizeField;
			ptr = *field - offset;
			if (size > 0) memcpy(*buf, ptr, size);
		}
	}
	if (listOut) executeTrans(1, cmd, 0, nullptr, listOut, offset, trans);
	return true;
}

__device__ volatile unsigned int _sentinelMapId;
__constant__ const sentinelMap *_sentinelDeviceMap[SENTINEL_DEVICEMAPS];
__device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength, sentinelInPtr *ptrsIn, sentinelOutPtr *ptrsOut) {
	const sentinelMap *map = _sentinelDeviceMap[_sentinelMapId++ % SENTINEL_DEVICEMAPS];
	if (!map)
		panic("sentinel: device map not defined. did you start sentinel?\n");

	// ATTACH
	long id = atomicAdd((int *)&map->setId, SENTINEL_MSGSIZE);
	sentinelCommand *cmd = (sentinelCommand *)&map->data[id % sizeof(map->data)];
	if (cmd->magic != SENTINEL_MAGIC)
		panic("bad sentinel magic");
	volatile long *control = &cmd->control; intptr_t offset = map->offset; char *trans = nullptr;
	mutexSpinLock(nullptr, control, SENTINELCONTROL_NORMAL, SENTINELCONTROL_DEVICE);

	// PREPARE
	char *data = cmd->data + ROUND8_(msgLength), *dataEnd = data + msg->size;
	sentinelOutPtr *listOut;
	if (((ptrsIn || ptrsOut) && !(data = preparePtrs(ptrsIn, ptrsOut, cmd, data, dataEnd, offset, listOut, trans))) ||
		(msg->prepare && !msg->prepare(msg, data, dataEnd, offset)))
		panic("msg too long");
	if (listOut) msg->flow &= SENTINELFLOW_TRAN;
	cmd->length = msgLength; memcpy(cmd->data, msg, msgLength);
	//printf("msg: %d[%d]'", msg->op, msgLength); for (int i = 0; i < msgLength; i++) printf("%02x", ((char *)msg)[i] & 0xff); printf("'\n");
	mutexSet(control, SENTINELCONTROL_DEVICERDY);

	// FLOW-WAIT
	if (msg->flow & SENTINELFLOW_WAIT) {
		mutexSpinLock(nullptr, control, SENTINELCONTROL_HOSTRDY, SENTINELCONTROL_DEVICEWAIT);
		cmd->length = msgLength; memcpy(msg, cmd->data, msgLength);
		if ((ptrsOut && !postfixPtrs(ptrsOut, cmd, offset, listOut, trans)) ||
			(msg->postfix && !msg->postfix(msg, offset)))
			panic("postfix error");
		mutexSet(control, SENTINELCONTROL_NORMAL);
	}
}

static __device__ void executeTrans(char id, sentinelCommand *cmd, int size, sentinelInPtr *listIn, sentinelOutPtr *listOut, intptr_t offset, char *&trans) {
	volatile long *control = &cmd->control;
	sentinelInPtr *i; sentinelOutPtr *o; char **field; char *data = cmd->data, *ptr = trans;
	switch (id) {
	case 0:
		*(int *)data = size;
		mutexSet(control, SENTINELCONTROL_TRANSSIZE);
		mutexSpinLock(nullptr, control, SENTINELCONTROL_TRANRDY, SENTINELCONTROL_TRANDONE);
		ptr = trans = *(char **)data;
		if (listIn)
			for (i = listIn; i; i = (sentinelInPtr *)i->unknown) {
				field = (char **)i->field;
				const char *v = (const char *)*field; int remain = i->size, length = 0;
				while (remain > 0) {
					length = cmd->length = remain > SENTINEL_MSGSIZE ? SENTINEL_MSGSIZE : remain;
					memcpy(data, (void *)v, length); remain -= length; v += length;
					mutexSet(control, SENTINELCONTROL_TRANSIN);
					mutexSpinLock(nullptr, control, SENTINELCONTROL_TRANRDY, SENTINELCONTROL_TRANDONE);
				}
				*field = ptr; ptr += i->size;
				i->unknown = nullptr;
			}
		if (listOut) {
			panic("listOut");
			for (o = listOut; o; o = (sentinelOutPtr *)o->unknown) {
				field = (char **)o->field;
				*field = ptr; ptr += o->size;
				o->unknown = nullptr;
			}
		}
		break;
	case 1:
		if (listOut)
			for (o = listOut; o; o = (sentinelOutPtr *)o->unknown) {
				field = (char **)o->field;
				const char *v = (const char *)*field; int remain = o->size, length = 0;
				while (remain > 0) {
					length = cmd->length = remain > SENTINEL_MSGSIZE ? SENTINEL_MSGSIZE : remain;
					memcpy((void *)v, data, length); remain -= length; v += length;
					mutexSet(control, SENTINELCONTROL_TRANSOUT);
					mutexSpinLock(nullptr, control, SENTINELCONTROL_TRANRDY, SENTINELCONTROL_TRANDONE);
				}
				*field = ptr; ptr += o->size;
				o->unknown = nullptr;
			}
		break;
	}
}

#endif

__END_DECLS;
