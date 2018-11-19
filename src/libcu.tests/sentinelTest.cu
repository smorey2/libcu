#include <stdiocu.h>
#include <stringcu.h>
#include <sentinel.h>
#include <assert.h>

enum {
	MODULE_SIMPLE = 500,
	MODULE_STRING,
	MODULE_CUSTOM,
	MODULE_COMPLEX,
	MODULE_JUMBOOUT,
};

struct module_simple {
	sentinelMessage base;
	int value;
	__device__ module_simple(bool wait, int value) : base(MODULE_SIMPLE, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE), value(value) { sentinelDeviceSend(&base, sizeof(module_simple)); }
	int rc;
};

struct module_string {
	sentinelMessage base;
	const char *str;
	__device__ module_string(bool wait, const char *str) : base(MODULE_STRING, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE, SENTINEL_CHUNK), str(str) { sentinelDeviceSend(&base, sizeof(module_string), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct module_custom {
	static __forceinline__ __device__ char *prepare(module_custom *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->str ? (int)strlen(t->str) + 1 : 0;
		char *str = data;
		char *end = data += strLength;
		if (end > dataEnd) return nullptr;
		memcpy(str, t->str, strLength);
		t->str = str + offset;
		return end;
	}
	sentinelMessage base;
	const char *str;
	__device__ module_custom(bool wait, const char *str) : base(MODULE_CUSTOM, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE, SENTINEL_CHUNK, SENTINELPREPARE(prepare)), str(str) { sentinelDeviceSend(&base, sizeof(module_custom)); }
	int rc;
};

struct module_complex {
	sentinelMessage base;
	const char *str;
	__device__ module_complex(bool wait, const char *str, char *ptr, size_t size) : base(MODULE_COMPLEX, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE, SENTINEL_CHUNK), str(str), ptr(ptr) { ptrsIn[1].size = size; sentinelDeviceSend(&base, sizeof(module_complex), ptrsIn); }
	int rc;
	void *ptr;
	sentinelInPtr ptrsIn[3] = {
		{ &str, -1 },
		{ &ptr, 0 },
		{ nullptr }
	};
};

//struct module_jumboout {
//	sentinelMessage base;
//	void *ptr; size_t size;
//	__device__ module_jumboout(char *ptr, size_t size) : base(MODULE_JUMBOOUT, SENTINELFLOW_WAIT, SENTINEL_CHUNK) { sentinelDeviceSend(&base, sizeof(module_jumboout)); }
//	int rc;
//	sentinelOutPtr ptrsOut[3] = {
//		{ &Str, -1 },
//		{ &Ptr, 0 },
//		nullptr
//	};
//};

bool sentinelModuleExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*, char*, char*, intptr_t)) {
	switch (data->op) {
	case MODULE_SIMPLE: { module_simple *msg = (module_simple *)data; msg->rc = msg->value; return true; }
	case MODULE_STRING: { module_string *msg = (module_string *)data; msg->rc = (int)strlen(msg->str); return true; }
	case MODULE_CUSTOM: { module_custom *msg = (module_custom *)data; msg->rc = (int)strlen(msg->str); return true; }
	case MODULE_COMPLEX: { module_complex *msg = (module_complex *)data; msg->rc = (int)strlen(msg->str); return true; }
	//case MODULE_JUMBOOUT: { module_jumboout *msg = (module_jumboout *)data; msg->rc = 0; return true; }
	}
	return false;
}
static sentinelExecutor _moduleExecutor = { nullptr, "module", sentinelModuleExecutor, nullptr };

static __global__ void g_sentinel_test1() {
	printf("sentinel_test1\n");

	//// SENTINELDEVICESEND ////
	//	extern __device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength);
	char complex[2048]; memset(complex, 1, sizeof(complex));
	//module_simple a0(true, 1); int a0a = a0.RC; assert(a0a == 1);
	//module_string a1(true, "test"); int a1a = a1.RC; assert(a1a == 4);
	//module_custom a2(true, "test"); int a2a = a2.RC; assert(a2a == 4);
	//module_complex a3(true, "test", complex, sizeof(complex)); int a3a = a3.RC; assert(a3a == 4);

	// JUMBO
	char jumbo[9046]; memset(jumbo, 2, sizeof(jumbo));
	module_complex b0(true, "test", jumbo, sizeof(jumbo));
	//int b1a = b0.RC; //assert(b1a == 4);

}

cudaError_t sentinel_test1() {
	sentinelRegisterExecutor(&_moduleExecutor);
	g_sentinel_test1<<<1, 1>>>(); return cudaDeviceSynchronize();
}

//// SENTINELDEFAULTEXECUTOR ////
//	extern bool sentinelDefaultExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*,char*,char*,intptr_t));

//// SENTINELSERVERINITIALIZE, SENTINELSERVERSHUTDOWN ////
//	extern void sentinelServerInitialize(sentinelExecutor *executor = nullptr, char *mapHostName = SENTINEL_NAME, bool hostSentinel = true, bool deviceSentinel = true);
//	extern void sentinelServerShutdown();

//// SENTINELDEVICESEND ////
//	extern __device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength);

//// SENTINELCLIENTINITIALIZE, SENTINELCLIENTSHUTDOWN ////
//	extern void sentinelClientInitialize(char *mapHostName = SENTINEL_NAME);
//	extern void sentinelClientShutdown();

//// SENTINELCLIENTSEND ////
//	extern void sentinelClientSend(sentinelMessage *msg, int msgLength);

//// SENTINELFINDEXECUTOR, SENTINELREGISTEREXECUTOR, SENTINELUNREGISTEREXECUTOR ////
//	extern sentinelExecutor *sentinelFindExecutor(const char *name, bool forDevice = true);
//	extern void sentinelRegisterExecutor(sentinelExecutor *exec, bool makeDefault = false, bool forDevice = true);
//	extern void sentinelUnregisterExecutor(sentinelExecutor *exec, bool forDevice = true);

//// SENTINELREGISTERFILEUTILS ////
//	extern void sentinelRegisterFileUtils();
