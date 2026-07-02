/* stdlib.h stub for ARM Cortex-M7 bare-metal (no newlib) */
#ifndef _STDLIB_H
#define _STDLIB_H

#include <stddef.h>

void  *malloc(size_t size);
void  *calloc(size_t nmemb, size_t size);
void  *realloc(void *ptr, size_t size);
void   free(void *ptr);
void   abort(void);
void   exit(int status);
int    abs(int j);
long   labs(long j);
long long llabs(long long j);
int    atoi(const char *nptr);
long   atol(const char *nptr);
long long atoll(const char *nptr);
double atof(const char *nptr);
long   strtol(const char *nptr, char **endptr, int base);
unsigned long strtoul(const char *nptr, char **endptr, int base);
long long strtoll(const char *nptr, char **endptr, int base);
unsigned long long strtoull(const char *nptr, char **endptr, int base);
double strtod(const char *nptr, char **endptr);

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

#endif /* _STDLIB_H */
