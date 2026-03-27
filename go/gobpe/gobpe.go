package gobpe

import (
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"regexp"
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
	merges map[[2]string]int
}

// NewBPETokenizer initialize BPETokenizer with training corpus and maxVocabSize
func NewBPETokenizer(corpus string, maxVocabSize int) *BPETokenizer {
	vocab := make(map[string]int, 2048)
	merges := make(map[[2]string]int)

	for i := 0; i < 256; i++ {
		vocab[string([]byte{byte(i)})] = i
	}

	corpusBytes := []byte(corpus)
	preTokens := gpt4Regex.FindAll(corpusBytes, -1)

	vocabFreq := make(map[string]int)
	for _, word := range preTokens {
		vocabFreq[string(word)]++
	}

	type wordEntry struct {
		tokens []string
		count  int
	}
	var words []*wordEntry

	for word, count := range vocabFreq {
		tokens := make([]string, 0, len(word))
		for i := 0; i < len(word); i++ {
			tokens = append(tokens, string([]byte{word[i]}))
		}
		words = append(words, &wordEntry{tokens: tokens, count: count})
	}

	for len(vocab) < maxVocabSize {
		pairsFreq := make(map[[2]string]int)
		var bestCount int
		var bestKey [2]string

		for _, entry := range words {
			for i := 0; i < len(entry.tokens)-1; i++ {
				key := [2]string{entry.tokens[i], entry.tokens[i+1]}
				pairsFreq[key] += entry.count

				if pairsFreq[key] > bestCount {
					bestCount = pairsFreq[key]
					bestKey = key
				}
			}
		}

		if bestCount < 1 {
			break
		}

		merges[bestKey] = len(vocab)
		vocab[bestKey[0]+bestKey[1]] = len(vocab)

		for _, entry := range words {
			entry.tokens = applyMerge(entry.tokens, bestKey)
		}
	}

	return &BPETokenizer{vocab: vocab, merges: merges}
}

// Encode processes new text using the learned merges
func (bpe *BPETokenizer) Encode(corpus string) []int {
	var result []int
	corpusBytes := []byte(corpus)
	preTokens := gpt4Regex.FindAll(corpusBytes, -1)

	for _, wordBytes := range preTokens {
		tokens := make([]string, 0, len(wordBytes))
		for i := 0; i < len(wordBytes); i++ {
			tokens = append(tokens, string([]byte{wordBytes[i]}))
		}

		for len(tokens) > 1 {
			minRank := math.MaxInt
			var bestPair [2]string
			var pairFound bool

			for i := 0; i < len(tokens)-1; i++ {
				pair := [2]string{tokens[i], tokens[i+1]}
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

			tokens = applyMerge(tokens, bestPair)
		}

		for _, t := range tokens {
			result = append(result, bpe.vocab[t])
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
		pair [2]string
		rank int
	}
	pairs := make([]rankPair, 0, len(bpe.merges))
	for p, r := range bpe.merges {
		pairs = append(pairs, rankPair{p, r})
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].rank < pairs[j].rank
	})

	mergesHex := make([][2]string, len(pairs))
	for i, p := range pairs {
		mergesHex[i] = [2]string{
			hex.EncodeToString([]byte(p.pair[0])),
			hex.EncodeToString([]byte(p.pair[1])),
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

	merges := make(map[[2]string]int, len(parsed.Merges))
	for i, hexPair := range parsed.Merges {
		left, err := hex.DecodeString(hexPair[0])
		if err != nil {
			return nil, err
		}
		right, err := hex.DecodeString(hexPair[1])
		if err != nil {
			return nil, err
		}
		merges[[2]string{string(left), string(right)}] = 256 + i
	}

	return &BPETokenizer{vocab: vocab, merges: merges}, nil
}

func applyMerge(tokens []string, pair [2]string) []string {
	if len(tokens) < 2 {
		return tokens
	}
	newTokens := make([]string, 0, len(tokens))
	i := 0

	for i < len(tokens) {
		if i < len(tokens)-1 && tokens[i] == pair[0] && tokens[i+1] == pair[1] {
			newTokens = append(newTokens, pair[0]+pair[1])
			i += 2
		} else {
			newTokens = append(newTokens, tokens[i])
			i++
		}
	}
	return newTokens
}
