package main

/*
#include <stdlib.h>
*/
import "C"
import (
	"sync"
	"unsafe"

	"github.com/venglov/go-bpe/go/gobpe"
)

var (
	mu        sync.Mutex
	instances = map[int]*gobpe.BPETokenizer{}
	nextID    = 1
)

func storeInstance(t *gobpe.BPETokenizer) C.int {
	mu.Lock()
	id := nextID
	nextID++
	instances[id] = t
	mu.Unlock()
	return C.int(id)
}

func getInstance(handle C.int) *gobpe.BPETokenizer {
	mu.Lock()
	t := instances[int(handle)]
	mu.Unlock()
	return t
}

//export BPE_Train
func BPE_Train(corpus *C.char, corpusLen C.int, maxVocabSize C.int) C.int {
	goCorpus := C.GoStringN(corpus, corpusLen)
	t := gobpe.NewBPETokenizer(goCorpus, int(maxVocabSize))
	return storeInstance(t)
}

//export BPE_Load
func BPE_Load(path *C.char) C.int {
	t, err := gobpe.LoadBPETokenizer(C.GoString(path))
	if err != nil {
		return -1
	}
	return storeInstance(t)
}

//export BPE_Save
func BPE_Save(handle C.int, path *C.char) C.int {
	t := getInstance(handle)
	if t == nil {
		return -1
	}
	if err := t.Save(C.GoString(path)); err != nil {
		return -1
	}
	return 0
}

//export BPE_Encode
func BPE_Encode(handle C.int, text *C.char, outLen *C.int) *C.int {
	t := getInstance(handle)
	if t == nil {
		*outLen = 0
		return nil
	}

	ids := t.Encode(C.GoString(text))
	*outLen = C.int(len(ids))
	if len(ids) == 0 {
		return nil
	}

	cArr := (*C.int)(C.malloc(C.size_t(len(ids)) * C.size_t(unsafe.Sizeof(C.int(0)))))
	slice := unsafe.Slice(cArr, len(ids))
	for i, v := range ids {
		slice[i] = C.int(v)
	}
	return cArr
}

//export BPE_Decode
func BPE_Decode(handle C.int, ids *C.int, idsLen C.int) *C.char {
	t := getInstance(handle)
	if t == nil {
		return C.CString("")
	}

	goIDs := make([]int, int(idsLen))
	if idsLen > 0 {
		cSlice := unsafe.Slice(ids, int(idsLen))
		for i, v := range cSlice {
			goIDs[i] = int(v)
		}
	}

	return C.CString(t.Decode(goIDs))
}

//export BPE_Free
func BPE_Free(handle C.int) {
	mu.Lock()
	delete(instances, int(handle))
	mu.Unlock()
}

//export BPE_FreeTokens
func BPE_FreeTokens(ptr *C.int) {
	if ptr != nil {
		C.free(unsafe.Pointer(ptr))
	}
}

//export BPE_FreeString
func BPE_FreeString(ptr *C.char) {
	if ptr != nil {
		C.free(unsafe.Pointer(ptr))
	}
}

func main() {}
