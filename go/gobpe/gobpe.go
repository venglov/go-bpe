package gobpe

import (
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"regexp"
	"slices"
	"sort"
)

var gpt4Regex = regexp.MustCompile(
	`(?i:'s|'t|'re|'ve|'m|'ll|'d)` + // contractions
		`|[^\r\n\p{L}\p{N}]?\p{L}+` + // words (with optional leading non-alphanum)
		`|\p{N}{1,3}` + // numbers (up to 3 digits)
		`| ?[^\s\p{L}\p{N}]+[\r\n]*` + // punctuation / special chars
		`|\s*[\r\n]+` + // newlines
		`|\s+(?:\S)` + // whitespace (not trailing)
		`|\s+`, // trailing whitespace
)

type BPETokenizer struct {
	vocab  map[string]int
	merges map[[2]int]int
}

type Node struct {
	next *Node
	prev *Node
	val  int
}

// NewBPETokenizer initialize BPETokenizer with training corpus and maxVocabSize
func NewBPETokenizer(corpus string, maxVocabSize int, specialTokens []string) *BPETokenizer {
	vocab := make(map[string]int, 2048)
	idToStr := make(map[int]string, 2048)
	merges := make(map[[2]int]int)
	pairToWordIdx := make(map[[2]int][]int)

	for i := 0; i < 256; i++ {
		s := string([]byte{byte(i)})
		vocab[s] = i
		idToStr[i] = s
	}

	corpusBytes := []byte(corpus)
	preTokens := gpt4Regex.FindAll(corpusBytes, -1)

	vocabFreq := make(map[string]int)
	for _, word := range preTokens {
		vocabFreq[string(word)]++
	}

	var add_new func(int, int)
	add_new = func(k1 int, k2 int) {
		newID := len(vocab)
		newStr := idToStr[k1] + idToStr[k2]
		vocab[newStr] = newID
		idToStr[newID] = newStr
	}

	type wordEntry struct {
		head  *Node
		count int
	}
	var words []*wordEntry

	for word, count := range vocabFreq {
		if slices.Contains(specialTokens, word) {
			continue
		}
		head := &Node{}
		tokens := head
		for i := 0; i < len(word); i++ {
			tokens.next = &Node{val: int(word[i])}
			tokens.next.prev = tokens
			tokens = tokens.next
		}
		words = append(words, &wordEntry{head: head, count: count})
	}

	pairsFreq := make(map[[2]int]int)

	for i, entry := range words {
		for token := entry.head.next; token != nil && token.next != nil; token = token.next {
			key := [2]int{token.val, token.next.val}
			pairsFreq[key] += entry.count
			pairToWordIdx[key] = append(pairToWordIdx[key], i)
		}
	}

	for len(vocab) < maxVocabSize-len(specialTokens) {
		var bestCount int
		var bestKey [2]int

		for pair, freq := range pairsFreq {
			if freq > bestCount {
				bestCount = freq
				bestKey = pair
			}
		}

		if bestCount < 1 {
			break
		}

		newID := len(vocab)
		merges[bestKey] = newID
		add_new(bestKey[0], bestKey[1])

		for _, wordIdx := range pairToWordIdx[bestKey] {
			entry := words[wordIdx]
			head := entry.head
			for node := head.next; node != nil && node.next != nil; {
				if node.val == bestKey[0] && node.next.val == bestKey[1] {
					prev := node.prev
					if prev != nil && prev.prev != nil {
						lkey := [2]int{prev.val, node.val}
						pairsFreq[lkey] -= entry.count

						lkey = [2]int{prev.val, newID}
						pairsFreq[lkey] += entry.count
						pairToWordIdx[lkey] = append(pairToWordIdx[lkey], wordIdx)
					}
					next := node.next.next
					if next != nil {
						rkey := [2]int{node.next.val, next.val}
						pairsFreq[rkey] -= entry.count

						rkey = [2]int{newID, next.val}
						pairsFreq[rkey] += entry.count
						pairToWordIdx[rkey] = append(pairToWordIdx[rkey], wordIdx)
					}

					node.val = newID
					node.next = node.next.next
					if node.next != nil {
						node.next.prev = node
					}

				} else {
					node = node.next
				}
			}
		}
		delete(pairsFreq, bestKey)
		delete(pairToWordIdx, bestKey)
	}

	for _, st := range specialTokens {
		vocab[st] = len(vocab)
	}

	return &BPETokenizer{vocab: vocab, merges: merges}
}

// Encode processes new text using the learned merges
func (bpe *BPETokenizer) Encode(corpus string) []int {
	var result []int
	corpusBytes := []byte(corpus)
	preTokens := gpt4Regex.FindAll(corpusBytes, -1)

	for _, wordBytes := range preTokens {
		head := &Node{}
		tokens := head
		for i := 0; i < len(wordBytes); i++ {
			tokens.next = &Node{val: int(wordBytes[i])}
			tokens = tokens.next
		}

		for {
			minRank := math.MaxInt
			var bestPair [2]int
			var pairFound bool

			for tokens := head.next; tokens != nil && tokens.next != nil; tokens = tokens.next {
				pair := [2]int{tokens.val, tokens.next.val}
				if rank, exists := bpe.merges[pair]; exists {
					if rank < minRank {
						minRank = rank
						bestPair = pair
						pairFound = true
					}
				}
			}

			if !pairFound {
				break
			}

			applyMerge(head, bestPair, int(bpe.merges[bestPair]))
		}

		for tokens := head.next; tokens != nil; tokens = tokens.next {
			result = append(result, int(tokens.val))
		}
	}

	return result
}

// Decode converts token IDs back to a string using the learned vocabulary
func (bpe *BPETokenizer) Decode(ids []int) string {
	idToToken := make(map[int]string, len(bpe.vocab))
	for token, id := range bpe.vocab {
		idToToken[id] = token
	}

	var result []byte
	for _, id := range ids {
		result = append(result, []byte(idToToken[id])...)
	}
	return string(result)
}

type bpeJSON struct {
	Vocab  map[string]int `json:"vocab"`
	Merges [][2]string    `json:"merges"`
}

// Save writes the tokenizer's vocabulary and merges to a JSON file.
// Vocab keys and merge pairs are hex-encoded to preserve arbitrary byte sequences.
func (bpe *BPETokenizer) Save(path string) error {
	type rankPair struct {
		pair [2]int
		rank int
	}
	pairs := make([]rankPair, 0, len(bpe.merges))
	for p, r := range bpe.merges {
		pairs = append(pairs, rankPair{p, r})
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].rank < pairs[j].rank
	})

	idToToken := make(map[int]string, len(bpe.vocab))
	for token, id := range bpe.vocab {
		idToToken[id] = token
	}

	mergesHex := make([][2]string, len(pairs))
	for i, p := range pairs {
		mergesHex[i] = [2]string{
			hex.EncodeToString([]byte(idToToken[int(p.pair[0])])),
			hex.EncodeToString([]byte(idToToken[int(p.pair[1])])),
		}
	}

	vocabHex := make(map[string]int, len(bpe.vocab))
	for k, v := range bpe.vocab {
		vocabHex[hex.EncodeToString([]byte(k))] = v
	}

	jsonBytes, err := json.Marshal(bpeJSON{Vocab: vocabHex, Merges: mergesHex})
	if err != nil {
		return err
	}
	return os.WriteFile(path, jsonBytes, 0644)
}

// LoadBPETokenizer loads a tokenizer from a JSON file previously created by Save.
func LoadBPETokenizer(path string) (*BPETokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var parsed bpeJSON
	if err := json.Unmarshal(data, &parsed); err != nil {
		return nil, err
	}

	vocab := make(map[string]int, len(parsed.Vocab))
	for hexKey, id := range parsed.Vocab {
		keyBytes, err := hex.DecodeString(hexKey)
		if err != nil {
			return nil, err
		}
		vocab[string(keyBytes)] = id
	}

	merges := make(map[[2]int]int, len(parsed.Merges))
	for i, hexPair := range parsed.Merges {
		left, err := hex.DecodeString(hexPair[0])
		if err != nil {
			return nil, err
		}
		right, err := hex.DecodeString(hexPair[1])
		if err != nil {
			return nil, err
		}
		leftID := int(vocab[string(left)])
		rightID := int(vocab[string(right)])
		merges[[2]int{leftID, rightID}] = 256 + i
	}

	return &BPETokenizer{vocab: vocab, merges: merges}, nil
}

func applyMerge(head *Node, pair [2]int, newID int) {
	for node := head.next; node != nil && node.next != nil; {
		if node.val == pair[0] && node.next.val == pair[1] {
			node.val = newID
			node.next = node.next.next
		} else {
			node = node.next
		}
	}
}
