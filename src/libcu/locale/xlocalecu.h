#ifndef _XLOCALE_H
#define _XLOCALE_H

typedef struct __locale_struct {
} *__locale_t;

/* POSIX 2008 makes locale_t official. */
typedef __locale_t locale_t;

#endif  /* _XLOCALE_H */