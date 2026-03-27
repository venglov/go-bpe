package gobpe

import (
	"os"
	"testing"
)

const smallCorpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"

func TestNewBPETokenizer_BaseVocab(t *testing.T) {
	bpe := NewBPETokenizer("", 256)
	// With an empty corpus and maxVocabSize==256, we get exactly the 256 byte tokens.
	// Tokenizing a single known byte should return its ASCII value.
	tokens := bpe.Encode("A")
	if len(tokens) != 1 {
		t.Fatalf("expected 1 token for single byte, got %d", len(tokens))
	}
	if tokens[0] != int('A') {
		t.Fatalf("expected token %d ('A'), got %d", int('A'), tokens[0])
	}
}

func TestTokenize_Deterministic(t *testing.T) {
	bpe := NewBPETokenizer(smallCorpus, 300)
	text := "low lower newest widest"
	first := bpe.Encode(text)
	second := bpe.Encode(text)
	if len(first) != len(second) {
		t.Fatalf("non-deterministic: lengths %d vs %d", len(first), len(second))
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("non-deterministic at index %d: %d vs %d", i, first[i], second[i])
		}
	}
}

func TestTokenize_CompressionOnTrainingData(t *testing.T) {
	bpe := NewBPETokenizer(smallCorpus, 300)
	tokens := bpe.Encode(smallCorpus)
	// A trained tokenizer must produce fewer tokens than bytes for a repetitive corpus.
	if len(tokens) >= len(smallCorpus) {
		t.Fatalf("expected compression: %d tokens >= %d bytes", len(tokens), len(smallCorpus))
	}
}

func TestTokenize_EmptyString(t *testing.T) {
	bpe := NewBPETokenizer(smallCorpus, 300)
	tokens := bpe.Encode("")
	if len(tokens) != 0 {
		t.Fatalf("expected empty result for empty input, got %v", tokens)
	}
}

func TestTokenize_UnseenText(t *testing.T) {
	bpe := NewBPETokenizer(smallCorpus, 300)
	// Text not in the training corpus should still tokenize (falls back to byte tokens).
	tokens := bpe.Encode("hello world")
	if len(tokens) == 0 {
		t.Fatal("expected non-empty token list for unseen text")
	}
}

func TestTokenize_NonASCII(t *testing.T) {
	bpe := NewBPETokenizer(smallCorpus, 300)
	tokens := bpe.Encode("café naïve")
	if len(tokens) == 0 {
		t.Fatal("expected non-empty token list for non-ASCII text")
	}
}

func TestApplyMerge_Basic(t *testing.T) {
	tokens := []string{"l", "o", "w"}
	pair := [2]string{"l", "o"}
	result := applyMerge(tokens, pair)
	if len(result) != 2 || result[0] != "lo" || result[1] != "w" {
		t.Fatalf("unexpected merge result: %v", result)
	}
}

func TestApplyMerge_NoMatch(t *testing.T) {
	tokens := []string{"l", "o", "w"}
	pair := [2]string{"x", "y"}
	result := applyMerge(tokens, pair)
	if len(result) != 3 {
		t.Fatalf("expected unchanged tokens, got %v", result)
	}
}

func TestApplyMerge_SingleToken(t *testing.T) {
	tokens := []string{"a"}
	result := applyMerge(tokens, [2]string{"a", "b"})
	if len(result) != 1 || result[0] != "a" {
		t.Fatalf("single token should be unchanged, got %v", result)
	}
}

func TestApplyMerge_ConsecutivePairs(t *testing.T) {
	// "a a a" with pair ("a","a") → "aa a" (non-overlapping, left-to-right)
	tokens := []string{"a", "a", "a"}
	result := applyMerge(tokens, [2]string{"a", "a"})
	if len(result) != 2 || result[0] != "aa" || result[1] != "a" {
		t.Fatalf("expected [aa a], got %v", result)
	}
}

func TestNewBPETokenizer_VocabGrows(t *testing.T) {
	// With maxVocabSize > 256, merges should be learned and vocab should grow.
	bpe256 := NewBPETokenizer(smallCorpus, 256)
	bpe300 := NewBPETokenizer(smallCorpus, 300)

	tokens256 := bpe256.Encode(smallCorpus)
	tokens300 := bpe300.Encode(smallCorpus)

	// More merges → fewer tokens
	if len(tokens300) >= len(tokens256) {
		t.Fatalf("expected more merges to reduce token count: %d (vocab 300) >= %d (vocab 256)",
			len(tokens300), len(tokens256))
	}
}

func TestDecode_RoundTrip(t *testing.T) {
	bpe := NewBPETokenizer(smallCorpus, 300)
	text := "low lower newest widest"
	tokens := bpe.Encode(text)
	decoded := bpe.Decode(tokens)
	if decoded != text {
		t.Fatalf("round-trip failed: got %q, want %q", decoded, text)
	}
}

func TestDecode_Empty(t *testing.T) {
	bpe := NewBPETokenizer(smallCorpus, 300)
	decoded := bpe.Decode([]int{})
	if decoded != "" {
		t.Fatalf("expected empty string, got %q", decoded)
	}
}

func TestSaveLoad_RoundTrip(t *testing.T) {
	bpe := NewBPETokenizer(smallCorpus, 300)
	text := "low lower newest widest"
	originalTokens := bpe.Encode(text)

	tmpFile := t.TempDir() + "/model.json"
	if err := bpe.Save(tmpFile); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	loaded, err := LoadBPETokenizer(tmpFile)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	loadedTokens := loaded.Encode(text)
	if len(originalTokens) != len(loadedTokens) {
		t.Fatalf("token count mismatch: %d vs %d", len(originalTokens), len(loadedTokens))
	}
	for i := range originalTokens {
		if originalTokens[i] != loadedTokens[i] {
			t.Fatalf("token mismatch at %d: %d vs %d", i, originalTokens[i], loadedTokens[i])
		}
	}

	decoded := loaded.Decode(loadedTokens)
	if decoded != text {
		t.Fatalf("decode round-trip failed: got %q, want %q", decoded, text)
	}
}

func TestLoad_NonExistentFile(t *testing.T) {
	_, err := LoadBPETokenizer("/nonexistent/path/model.json")
	if err == nil {
		t.Fatal("expected error for nonexistent file")
	}
}

func TestTokenize_WithVerdictCorpus(t *testing.T) {
	data, err := os.ReadFile("../the-verdict.txt")
	if err != nil {
		t.Skip("the-verdict.txt not found, skipping integration test")
	}
	corpus := string(data)
	bpe := NewBPETokenizer(corpus, 512)

	tokens := bpe.Encode(corpus)
	if len(tokens) == 0 {
		t.Fatal("expected non-empty token list")
	}
	ratio := float64(len(corpus)) / float64(len(tokens))
	if ratio < 1.5 {
		t.Fatalf("expected compression ratio >= 1.5, got %.2f", ratio)
	}
	t.Logf("Corpus: %d bytes → %d tokens (ratio %.2fx)", len(corpus), len(tokens), ratio)
}
