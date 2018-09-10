#include <limits.h>

#define	__STRUCT_COMMON

struct __xlocale_st_runelocale {
	__STRUCT_COMMON;
	char __ctype_encoding[ENCODING_LEN + 1];
	int __mb_cur_max;
	int __mb_sb_limit;
	size_t(*__mbrtowc)(wchar_t *__restrict, const char *__restrict, size_t, __darwin_mbstate_t* __restrict, struct _xlocale *);
	int(*__mbsinit)(const __darwin_mbstate_t *, struct _xlocale *);
	size_t(*__mbsnrtowcs)(wchar_t * __restrict, const char ** __restrict, size_t, size_t, __darwin_mbstate_t* __restrict, struct _xlocale *);
	size_t(*__wcrtomb)(char * __restrict, wchar_t, __darwin_mbstate_t* __restrict, struct _xlocale *);
	size_t(*__wcsnrtombs)(char * __restrict, const wchar_t ** __restrict, size_t, size_t, __darwin_mbstate_t* __restrict, struct _xlocale *);
	int __datasize;
};

struct _xlocale {
	__STRUCT_COMMON;
	/* 10 independent mbstate_t buffers! */
	__darwin_mbstate_t __mbs_mblen;
	__darwin_mbstate_t __mbs_mbrlen;
	__darwin_mbstate_t __mbs_mbrtowc;
	__darwin_mbstate_t __mbs_mbsnrtowcs;
	__darwin_mbstate_t __mbs_mbsrtowcs;
	__darwin_mbstate_t __mbs_mbtowc;
	__darwin_mbstate_t __mbs_wcrtomb;
	__darwin_mbstate_t __mbs_wcsnrtombs;
	__darwin_mbstate_t __mbs_wcsrtombs;
	__darwin_mbstate_t __mbs_wctomb;
	pthread_lock_t __lock;
	/* magic (Here up to the end is copied when duplicating locale_t's) */
	int64_t __magic;
};

#define NORMALIZE_LOCALE(x)	if (!(x)) { \
					(x) = _c_locale; \
				} else if ((x) == LC_GLOBAL_LOCALE) { \
					(x) = &__global_locale; \
				}

__BEGIN_DECLS;
static inline locale_t __current_locale() { return &__global_locale; }
__END_DECLS;

#endif /* _XLOCALE_PRIVATE_H_ */