// tclHash.c --
//
//	Implementation of in-memory hash tables for Tcl and Tcl-based applications.
//
// Copyright 1991 Regents of the University of California
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without
// fee is hereby granted, provided that this copyright notice appears in all copies.  The University of California
// makes no representations about the suitability of this software for any purpose.  It is provided "as is" without
// express or implied warranty.

#include "tclInt.h"

// When there are this many entries per bucket, on average, rebuild the hash table to make it larger.
#define REBUILD_MULTIPLIER	3

// The following macro takes a preliminary integer hash value and produces an index into a hash tables bucket list.  The idea is
// to make it so that preliminary values that are arbitrarily similar will end up in different buckets.  The hash function was taken
// from a random-number generator.
#define RANDOM_INDEX(tablePtr, i) (int)(((((long)(i))*1103515245) >> (tablePtr)->downShift) & (tablePtr)->mask)

// Procedure prototypes for static procedures in this file:
static __device__ Tcl_HashEntry *ArrayFind(Tcl_HashTable *tablePtr, const char *key);
static __device__ Tcl_HashEntry *ArrayCreate(Tcl_HashTable *tablePtr, const char *key, int *newPtr);
static __device__ Tcl_HashEntry *BogusFind(Tcl_HashTable *tablePtr, const char *key);
static __device__ Tcl_HashEntry *BogusCreate(Tcl_HashTable *tablePtr, const char *key, int *newPtr);
static __device__ unsigned int HashString(const char *string);
static __device__ void RebuildTable(Tcl_HashTable *tablePtr);
static __device__ Tcl_HashEntry *StringFind(Tcl_HashTable *tablePtr, const char *key);
static __device__ Tcl_HashEntry *StringCreate(Tcl_HashTable *tablePtr, const char *key, int *newPtr);
static __device__ Tcl_HashEntry *OneWordFind(Tcl_HashTable *tablePtr, const char *key);
static __device__ Tcl_HashEntry *OneWordCreate(Tcl_HashTable *tablePtr, const char *key, int *newPtr);

/*
*----------------------------------------------------------------------
*
* Tcl_InitHashTable --
*	Given storage for a hash table, set up the fields to prepare the hash table for use.
*
* Results:
*	None.
*
* Side effects:
*	TablePtr is now ready to be passed to Tcl_FindHashEntry and Tcl_CreateHashEntry.
*
*----------------------------------------------------------------------
*/
__device__ void Tcl_InitHashTable(register Tcl_HashTable *tablePtr, int keyType)
{
	tablePtr->buckets = tablePtr->staticBuckets;
	tablePtr->staticBuckets[0] = tablePtr->staticBuckets[1] = 0;
	tablePtr->staticBuckets[2] = tablePtr->staticBuckets[3] = 0;
	tablePtr->numBuckets = TCL_SMALL_HASH_TABLE;
	tablePtr->numEntries = 0;
	tablePtr->rebuildSize = TCL_SMALL_HASH_TABLE*REBUILD_MULTIPLIER;
	tablePtr->downShift = 28;
	tablePtr->mask = 3;
	tablePtr->keyType = keyType;
	if (keyType == TCL_STRING_KEYS) {
		tablePtr->findProc = StringFind;
		tablePtr->createProc = StringCreate;
	} else if (keyType == TCL_ONE_WORD_KEYS) {
		tablePtr->findProc = OneWordFind;
		tablePtr->createProc = OneWordCreate;
	} else {
		tablePtr->findProc = ArrayFind;
		tablePtr->createProc = ArrayCreate;
	};
}

/*
*----------------------------------------------------------------------
*
* Tcl_DeleteHashEntry --
*	Remove a single entry from a hash table.
*
* Results:
*	None.
*
* Side effects:
*	The entry given by entryPtr is deleted from its table and should never again be used by the caller.  It is up to the
*	caller to free the clientData field of the entry, if that is relevant.
*
*----------------------------------------------------------------------
*/
__device__ void Tcl_DeleteHashEntry(Tcl_HashEntry *entryPtr)
{
	if (*entryPtr->bucketPtr == entryPtr) {
		*entryPtr->bucketPtr = entryPtr->nextPtr;
	} else {
		for (register Tcl_HashEntry *prevPtr = *entryPtr->bucketPtr; ; prevPtr = prevPtr->nextPtr) {
			if (prevPtr == NULL) {
				panic("malformed bucket chain in Tcl_DeleteHashEntry");
			}
			if (prevPtr->nextPtr == entryPtr) {
				prevPtr->nextPtr = entryPtr->nextPtr;
				break;
			}
		}
	}
	entryPtr->tablePtr->numEntries--;
	_freeFast((char *)entryPtr);
}

/*
*----------------------------------------------------------------------
*
* Tcl_DeleteHashTable --
*	Free up everything associated with a hash table except for the record for the table itself.
*
* Results:
*	None.
*
* Side effects:
*	The hash table is no longer useable.
*
*----------------------------------------------------------------------
*/
__device__ void Tcl_DeleteHashTable(register Tcl_HashTable *tablePtr)
{
	// Free up all the entries in the table.
	for (int i = 0; i < tablePtr->numBuckets; i++) {
		register Tcl_HashEntry *hPtr = tablePtr->buckets[i];
		while (hPtr != NULL) {
			register Tcl_HashEntry *nextPtr = hPtr->nextPtr;
			_freeFast((char *)hPtr);
			hPtr = nextPtr;
		}
	}
	// Free up the bucket array, if it was dynamically allocated.
	if (tablePtr->buckets != tablePtr->staticBuckets) {
		_freeFast((char *)tablePtr->buckets);
	}
	// Arrange for panics if the table is used again without re-initialization.
	tablePtr->findProc = BogusFind;
	tablePtr->createProc = BogusCreate;
}

/*
*----------------------------------------------------------------------
*
* Tcl_FirstHashEntry --
*	Locate the first entry in a hash table and set up a record that can be used to step through all the remaining entries of the table.
*
* Results:
*	The return value is a pointer to the first entry in tablePtr, or NULL if tablePtr has no entries in it.  The memory at
*	*searchPtr is initialized so that subsequent calls to Tcl_NextHashEntry will return all of the entries in the table, one at a time.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ Tcl_HashEntry *Tcl_FirstHashEntry(Tcl_HashTable *tablePtr, Tcl_HashSearch *searchPtr)
{
	searchPtr->tablePtr = tablePtr;
	searchPtr->nextIndex = 0;
	searchPtr->nextEntryPtr = NULL;
	return Tcl_NextHashEntry(searchPtr);
}

/*
*----------------------------------------------------------------------
*
* Tcl_NextHashEntry --
*	Once a hash table enumeration has been initiated by calling Tcl_FirstHashEntry, this procedure may be called to return
*	successive elements of the table.
*
* Results:
*	The return value is the next entry in the hash table being enumerated, or NULL if the end of the table is reached.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ Tcl_HashEntry *Tcl_NextHashEntry(register Tcl_HashSearch *searchPtr)
{
	while (searchPtr->nextEntryPtr == NULL) {
		if (searchPtr->nextIndex >= searchPtr->tablePtr->numBuckets) {
			return NULL;
		}
		searchPtr->nextEntryPtr = searchPtr->tablePtr->buckets[searchPtr->nextIndex];
		searchPtr->nextIndex++;
	}
	Tcl_HashEntry *hPtr = searchPtr->nextEntryPtr;
	searchPtr->nextEntryPtr = hPtr->nextPtr;
	return hPtr;
}

/*
*----------------------------------------------------------------------
*
* Tcl_HashStats --
*	Return statistics describing the layout of the hash table in its hash buckets.
*
* Results:
*	The return value is a malloc-ed string containing information about tablePtr.  It is the caller's responsibility to free this string.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ char *Tcl_HashStats(Tcl_HashTable *tablePtr)
{
#define NUM_COUNTERS 10
	int i;
	// Compute a histogram of bucket usage.
	int count[NUM_COUNTERS];
	for (i = 0; i < NUM_COUNTERS; i++) {
		count[i] = 0;
	}
	int overflow = 0;
	double average = 0.0;
	for (i = 0; i < tablePtr->numBuckets; i++) {
		int j = 0;
		for (register Tcl_HashEntry *hPtr = tablePtr->buckets[i]; hPtr != NULL; hPtr = hPtr->nextPtr) {
			j++;
		}
		if (j < NUM_COUNTERS) {
			count[j]++;
		} else {
			overflow++;
		}
		double tmp = j;
		average += (tmp+1.0)*(tmp/tablePtr->numEntries)/2.0;
	}
	// Print out the histogram and a few other pieces of information.
	char *result = (char *)_allocFast((unsigned)((NUM_COUNTERS*60) + 300));
	sprintf(result, "%d entries in table, %d buckets\n", tablePtr->numEntries, tablePtr->numBuckets);
	char *p = result + strlen(result);
	for (i = 0; i < NUM_COUNTERS; i++) {
		sprintf(p, "number of buckets with %d entries: %d\n", i, count[i]);
		p += strlen(p);
	}
	sprintf(p, "number of buckets with more %d or more entries: %d\n", NUM_COUNTERS, overflow);
	p += strlen(p);
	sprintf(p, "average search distance for entry: %.1f", average);
	return result;
}

/*
*----------------------------------------------------------------------
*
* HashString --
*	Compute a one-word summary of a text string, which can be used to generate a hash index.
*
* Results:
*	The return value is a one-word summary of the information in string.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
static __device__ unsigned int HashString(register const char *string)
{
	// I tried a zillion different hash functions and asked many other people for advice.  Many people had their own favorite functions,
	// all different, but no-one had much idea why they were good ones. I chose the one below (multiply by 9 and add new character)
	// because of the following reasons:
	//
	// 1. Multiplying by 10 is perfect for keys that are decimal strings, and multiplying by 9 is just about as good.
	// 2. Times-9 is (shift-left-3) plus (old).  This means that each character's bits hang around in the low-order bits of the
	//    hash value for ever, plus they spread fairly rapidly up to the high-order bits to fill out the hash value.  This seems
	//    works well both for decimal and non-decimal strings.
	register unsigned int result = 0;
	while (true) {
		register int c = *string;
		string++;
		if (c == 0) {
			break;
		}
		result += (result<<3) + c;
	}
	return result;
}

/*
*----------------------------------------------------------------------
*
* StringFind --
*	Given a hash table with string keys, and a string key, find the entry with a matching key.
*
* Results:
*	The return value is a token for the matching entry in the hash table, or NULL if there was no matching entry.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
static __device__ Tcl_HashEntry *StringFind(Tcl_HashTable *tablePtr, const char *key)
{
	int index = HashString(key) & tablePtr->mask;
	// Search all of the entries in the appropriate bucket.
	register const char *p1, *p2;
	for (register Tcl_HashEntry *hPtr = tablePtr->buckets[index]; hPtr != NULL; hPtr = hPtr->nextPtr) {
		for (p1 = key, p2 = hPtr->key.string; ; p1++, p2++) {
			if (*p1 != *p2) {
				break;
			}
			if (*p1 == '\0') {
				return hPtr;
			}
		}
	}
	return NULL;
}

/*
*----------------------------------------------------------------------
*
* StringCreate --
*	Given a hash table with string keys, and a string key, find the entry with a matching key.  If there is no matching entry,
*	then create a new entry that does match.
*
* Results:
*	The return value is a pointer to the matching entry.  If this is a newly-created entry, then *newPtr will be set to a non-zero
*	value;  otherwise *newPtr will be set to 0.  If this is a new entry the value stored in the entry will initially be 0.
*
* Side effects:
*	A new entry may be added to the hash table.
*
*----------------------------------------------------------------------
*/
static __device__ Tcl_HashEntry *StringCreate(Tcl_HashTable *tablePtr, const char *key, int *newPtr)
{
	register Tcl_HashEntry *hPtr;
	int index = HashString(key) & tablePtr->mask;
	// Search all of the entries in this bucket.
	register const char *p1, *p2;
	for (hPtr = tablePtr->buckets[index]; hPtr != NULL; hPtr = hPtr->nextPtr) {
		for (p1 = key, p2 = hPtr->key.string; ; p1++, p2++) {
			if (*p1 != *p2) {
				break;
			}
			if (*p1 == '\0') {
				*newPtr = 0;
				return hPtr;
			}
		}
	}
	// Entry not found.  Add a new one to the bucket.
	*newPtr = 1;
	hPtr = (Tcl_HashEntry *)_allocFast((unsigned)(sizeof(Tcl_HashEntry) + strlen(key) - (sizeof(hPtr->key) -1)));
	hPtr->tablePtr = tablePtr;
	hPtr->bucketPtr = &(tablePtr->buckets[index]);
	hPtr->nextPtr = *hPtr->bucketPtr;
	hPtr->clientData = 0;
	strcpy(hPtr->key.string, key);
	*hPtr->bucketPtr = hPtr;
	tablePtr->numEntries++;
	// If the table has exceeded a decent size, rebuild it with many more buckets.
	if (tablePtr->numEntries >= tablePtr->rebuildSize) {
		RebuildTable(tablePtr);
	}
	return hPtr;
}

/*
*----------------------------------------------------------------------
*
* OneWordFind --
*	Given a hash table with one-word keys, and a one-word key, find the entry with a matching key.
*
* Results:
*	The return value is a token for the matching entry in the hash table, or NULL if there was no matching entry.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
static __device__ Tcl_HashEntry *OneWordFind(Tcl_HashTable *tablePtr, register const char *key)
{
	int index = RANDOM_INDEX(tablePtr, key);
	// Search all of the entries in the appropriate bucket.
	for (register Tcl_HashEntry *hPtr = tablePtr->buckets[index]; hPtr != NULL; hPtr = hPtr->nextPtr) {
		if (hPtr->key.oneWordValue == key) {
			return hPtr;
		}
	}
	return NULL;
}

/*
*----------------------------------------------------------------------
*
* OneWordCreate --
*	Given a hash table with one-word keys, and a one-word key, find the entry with a matching key.  If there is no matching entry,
*	then create a new entry that does match.
*
* Results:
*	The return value is a pointer to the matching entry.  If this is a newly-created entry, then *newPtr will be set to a non-zero
*	value;  otherwise *newPtr will be set to 0.  If this is a new entry the value stored in the entry will initially be 0.
*
* Side effects:
*	A new entry may be added to the hash table.
*
*----------------------------------------------------------------------
*/
static __device__ Tcl_HashEntry *OneWordCreate(Tcl_HashTable *tablePtr, register const char *key, int *newPtr)
{
	register Tcl_HashEntry *hPtr;
	int index = RANDOM_INDEX(tablePtr, key);
	// Search all of the entries in this bucket.
	for (hPtr = tablePtr->buckets[index]; hPtr != NULL; hPtr = hPtr->nextPtr) {
		if (hPtr->key.oneWordValue == key) {
			*newPtr = 0;
			return hPtr;
		}
	}
	// Entry not found.  Add a new one to the bucket.
	*newPtr = 1;
	hPtr = (Tcl_HashEntry *)_allocFast(sizeof(Tcl_HashEntry));
	hPtr->tablePtr = tablePtr;
	hPtr->bucketPtr = &(tablePtr->buckets[index]);
	hPtr->nextPtr = *hPtr->bucketPtr;
	hPtr->clientData = 0;
	hPtr->key.oneWordValue = key;
	*hPtr->bucketPtr = hPtr;
	tablePtr->numEntries++;
	// If the table has exceeded a decent size, rebuild it with many more buckets.
	if (tablePtr->numEntries >= tablePtr->rebuildSize) {
		RebuildTable(tablePtr);
	}
	return hPtr;
}

/*
*----------------------------------------------------------------------
*
* ArrayFind --
*
*	Given a hash table with array-of-int keys, and a key, find the entry with a matching key.
*
* Results:
*	The return value is a token for the matching entry in the hash table, or NULL if there was no matching entry.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
static __device__ Tcl_HashEntry *ArrayFind(Tcl_HashTable *tablePtr, const char *key)
{
	int *arrayPtr = (int *) key;
	register int *iPtr1, *iPtr2;
	int index, count;
	for (index = 0, count = tablePtr->keyType, iPtr1 = arrayPtr; count > 0; count--, iPtr1++) {
		index += *iPtr1;
	}
	index = RANDOM_INDEX(tablePtr, index);
	// Search all of the entries in the appropriate bucket.
	for (register Tcl_HashEntry *hPtr = tablePtr->buckets[index]; hPtr != NULL; hPtr = hPtr->nextPtr) {
		for (iPtr1 = arrayPtr, iPtr2 = hPtr->key.words, count = tablePtr->keyType; ; count--, iPtr1++, iPtr2++) {
			if (count == 0) {
				return hPtr;
			}
			if (*iPtr1 != *iPtr2) {
				break;
			}
		}
	}
	return NULL;
}

/*
*----------------------------------------------------------------------
*
* ArrayCreate --
*	Given a hash table with one-word keys, and a one-word key, find the entry with a matching key.  If there is no matching entry,
*	then create a new entry that does match.
*
* Results:
*	The return value is a pointer to the matching entry.  If this is a newly-created entry, then *newPtr will be set to a non-zero
*	value;  otherwise *newPtr will be set to 0.  If this is a new entry the value stored in the entry will initially be 0.
*
* Side effects:
*	A new entry may be added to the hash table.
*
*----------------------------------------------------------------------
*/
static __device__ Tcl_HashEntry *ArrayCreate(Tcl_HashTable *tablePtr, register const char *key, int *newPtr)
{
	register Tcl_HashEntry *hPtr;
	int *arrayPtr = (int *) key;
	register int *iPtr1, *iPtr2;
	int index, count;
	for (index = 0, count = tablePtr->keyType, iPtr1 = arrayPtr; count > 0; count--, iPtr1++) {
		index += *iPtr1;
	}
	index = RANDOM_INDEX(tablePtr, index);
	// Search all of the entries in the appropriate bucket.
	for (hPtr = tablePtr->buckets[index]; hPtr != NULL; hPtr = hPtr->nextPtr) {
		for (iPtr1 = arrayPtr, iPtr2 = hPtr->key.words, count = tablePtr->keyType; ; count--, iPtr1++, iPtr2++) {
			if (count == 0) {
				*newPtr = 0;
				return hPtr;
			}
			if (*iPtr1 != *iPtr2) {
				break;
			}
		}
	}
	// Entry not found.  Add a new one to the bucket.
	*newPtr = 1;
	hPtr = (Tcl_HashEntry *)_allocFast((unsigned) (sizeof(Tcl_HashEntry) + (tablePtr->keyType*sizeof(int)) - 4));
	hPtr->tablePtr = tablePtr;
	hPtr->bucketPtr = &(tablePtr->buckets[index]);
	hPtr->nextPtr = *hPtr->bucketPtr;
	hPtr->clientData = 0;
	for (iPtr1 = arrayPtr, iPtr2 = hPtr->key.words, count = tablePtr->keyType; count > 0; count--, iPtr1++, iPtr2++) {
		*iPtr2 = *iPtr1;
	}
	*hPtr->bucketPtr = hPtr;
	tablePtr->numEntries++;
	// If the table has exceeded a decent size, rebuild it with many more buckets.
	if (tablePtr->numEntries >= tablePtr->rebuildSize) {
		RebuildTable(tablePtr);
	}
	return hPtr;
}

/*
*----------------------------------------------------------------------
*
* BogusFind --
*	This procedure is invoked when an Tcl_FindHashEntry is called on a table that has been deleted.
*
* Results:
*	If panic returns (which it shouldn't) this procedure returns NULL.
*
* Side effects:
*	Generates a panic.
*
*----------------------------------------------------------------------
*/
static __device__ Tcl_HashEntry *BogusFind(Tcl_HashTable *tablePtr, const char *key)
{
	panic("called Tcl_FindHashEntry on deleted table");
	return NULL;
}

/*
*----------------------------------------------------------------------
*
* BogusCreate --
*	This procedure is invoked when an Tcl_CreateHashEntry is called on a table that has been deleted.
*
* Results:
*	If panic returns (which it shouldn't) this procedure returns NULL.
*
* Side effects:
*	Generates a panic.
*
*----------------------------------------------------------------------
*/
static __device__ Tcl_HashEntry *BogusCreate(Tcl_HashTable *tablePtr, const char *key, int *newPtr)
{
	panic("called Tcl_CreateHashEntry on deleted table");
	return NULL;
}

/*
*----------------------------------------------------------------------
*
* RebuildTable --
*	This procedure is invoked when the ratio of entries to hash buckets becomes too large.  It creates a new table with a
*	larger bucket array and moves all of the entries into the new table.
*
* Results:
*	None.
*
* Side effects:
*	Memory gets reallocated and entries get re-hashed to new buckets.
*
*----------------------------------------------------------------------
*/
static __device__ void RebuildTable(register Tcl_HashTable *tablePtr)
{
	int oldSize = tablePtr->numBuckets;
	Tcl_HashEntry **oldBuckets = tablePtr->buckets;
	// Allocate and initialize the new bucket array, and set up hashing constants for new array size.
	tablePtr->numBuckets *= 4;
	tablePtr->buckets = (Tcl_HashEntry **)_allocFast((unsigned)(tablePtr->numBuckets * sizeof(Tcl_HashEntry *)));
	int count;
	register Tcl_HashEntry **newChainPtr;
	for (count = tablePtr->numBuckets, newChainPtr = tablePtr->buckets; count > 0; count--, newChainPtr++) {
		*newChainPtr = NULL;
	}
	tablePtr->rebuildSize *= 4;
	tablePtr->downShift -= 2;
	tablePtr->mask = (tablePtr->mask << 2) + 3;
	// Rehash all of the existing entries into the new bucket array.
	register Tcl_HashEntry **oldChainPtr;
	for (oldChainPtr = oldBuckets; oldSize > 0; oldSize--, oldChainPtr++) {
		for (register Tcl_HashEntry *hPtr = *oldChainPtr; hPtr != NULL; hPtr = *oldChainPtr) {
			*oldChainPtr = hPtr->nextPtr;
			int index;
			if (tablePtr->keyType == TCL_STRING_KEYS) {
				index = HashString(hPtr->key.string) & tablePtr->mask;
			} else if (tablePtr->keyType == TCL_ONE_WORD_KEYS) {
				index = RANDOM_INDEX(tablePtr, hPtr->key.oneWordValue);
			} else {
				register int *iPtr;
				for (index = 0, count = tablePtr->keyType, iPtr = hPtr->key.words; count > 0; count--, iPtr++) {
					index += *iPtr;
				}
				index = RANDOM_INDEX(tablePtr, index);
			}
			hPtr->bucketPtr = &(tablePtr->buckets[index]);
			hPtr->nextPtr = *hPtr->bucketPtr;
			*hPtr->bucketPtr = hPtr;
		}
	}
	// Free up the old bucket array, if it was dynamically allocated.
	if (oldBuckets != tablePtr->staticBuckets) {
		_freeFast((char *)oldBuckets);
	}
}
