package gobpe

import (
	"os"
	"strings"
	"testing"
)

const benchCorpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"

// --- Training benchmarks ---

func BenchmarkNewBPETokenizer_SmallCorpus_256(b *testing.B) {
	for b.Loop() {
		NewBPETokenizer(benchCorpus, 256, []string{})
	}
}

func BenchmarkNewBPETokenizer_SmallCorpus_300(b *testing.B) {
	for b.Loop() {
		NewBPETokenizer(benchCorpus, 300, []string{})
	}
}

func BenchmarkNewBPETokenizer_SmallCorpus_300_WithSpecialTokens(b *testing.B) {
	specials := []string{"<|endoftext|>", "<|startoftext|>", "<|pad|>"}
	for b.Loop() {
		NewBPETokenizer(benchCorpus, 300, specials)
	}
}

func BenchmarkNewBPETokenizer_Verdict_512(b *testing.B) {
	data, err := os.ReadFile("../the-verdict.txt")
	if err != nil {
		b.Skip("the-verdict.txt not found")
	}
	corpus := string(data)
	b.ResetTimer()
	for b.Loop() {
		NewBPETokenizer(corpus, 512, []string{})
	}
}

func BenchmarkNewBPETokenizer_Verdict_1024(b *testing.B) {
	data, err := os.ReadFile("../the-verdict.txt")
	if err != nil {
		b.Skip("the-verdict.txt not found")
	}
	corpus := string(data)
	b.ResetTimer()
	for b.Loop() {
		NewBPETokenizer(corpus, 1024, []string{})
	}
}

// --- Encode benchmarks ---

func BenchmarkEncode_SmallText(b *testing.B) {
	bpe := NewBPETokenizer(benchCorpus, 300, []string{})
	text := "low lower newest widest"
	b.ResetTimer()
	for b.Loop() {
		bpe.Encode(text)
	}
}

func BenchmarkEncode_Verdict(b *testing.B) {
	data, err := os.ReadFile("../the-verdict.txt")
	if err != nil {
		b.Skip("the-verdict.txt not found")
	}
	corpus := string(data)
	bpe := NewBPETokenizer(corpus, 512, []string{})
	b.ResetTimer()
	for b.Loop() {
		bpe.Encode(corpus)
	}
}

func BenchmarkEncode_LargeInput(b *testing.B) {
	data, err := os.ReadFile("../the-verdict.txt")
	if err != nil {
		b.Skip("the-verdict.txt not found")
	}
	corpus := string(data)
	bpe := NewBPETokenizer(corpus, 512, []string{})
	// 10x the corpus to stress-test encoding
	largeInput := strings.Repeat(corpus, 10)
	b.ResetTimer()
	for b.Loop() {
		bpe.Encode(largeInput)
	}
}

// --- Decode benchmarks ---

func BenchmarkDecode_SmallText(b *testing.B) {
	bpe := NewBPETokenizer(benchCorpus, 300, []string{})
	tokens := bpe.Encode("low lower newest widest")
	b.ResetTimer()
	for b.Loop() {
		bpe.Decode(tokens)
	}
}

func BenchmarkDecode_Verdict(b *testing.B) {
	data, err := os.ReadFile("../the-verdict.txt")
	if err != nil {
		b.Skip("the-verdict.txt not found")
	}
	corpus := string(data)
	bpe := NewBPETokenizer(corpus, 512, []string{})
	tokens := bpe.Encode(corpus)
	b.ResetTimer()
	for b.Loop() {
		bpe.Decode(tokens)
	}
}

// --- Save/Load benchmarks ---

func BenchmarkSave(b *testing.B) {
	data, err := os.ReadFile("../the-verdict.txt")
	if err != nil {
		b.Skip("the-verdict.txt not found")
	}
	bpe := NewBPETokenizer(string(data), 512, []string{})
	path := b.TempDir() + "/model.json"
	b.ResetTimer()
	for b.Loop() {
		bpe.Save(path)
	}
}

func BenchmarkLoad(b *testing.B) {
	data, err := os.ReadFile("../the-verdict.txt")
	if err != nil {
		b.Skip("the-verdict.txt not found")
	}
	bpe := NewBPETokenizer(string(data), 512, []string{})
	path := b.TempDir() + "/model.json"
	bpe.Save(path)
	b.ResetTimer()
	for b.Loop() {
		LoadBPETokenizer(path)
	}
}
