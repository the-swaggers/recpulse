import sys
import os
sys.path.insert(0, '.')
import recpulse_cuda as rp

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name}")
        failed += 1

print("=" * 60)
print("BPE Tokenizer Tests")
print("=" * 60)

print("\n--- Creation ---")

tok = rp.Tokenizer()
check("create tokenizer", tok is not None)

print("\n--- Training (basic) ---")

text = "aaabdaaabac"
tok.train(text, vocab_size=260)
check("vocab_size > 256", tok.vocab_size > 256)

print("\n--- Encode / Decode roundtrip ---")

ids = tok.encode(text)
check("encode returns list", isinstance(ids, list))
check("encode non-empty", len(ids) > 0)
check("encode shorter than input", len(ids) <= len(text))

decoded = tok.decode(ids)
check("decode roundtrip", decoded == text)

print("\n--- BPE compression ---")

text2 = "the cat sat on the mat the cat sat on the mat " * 10
tok2 = rp.Tokenizer()
tok2.train(text2, vocab_size=300)
ids2 = tok2.encode(text2)
check("BPE compresses", len(ids2) < len(text2))
check(f"compression ratio ({len(ids2)}/{len(text2)} = {len(ids2)/len(text2):.2f})", len(ids2) < len(text2) * 0.7)
decoded2 = tok2.decode(ids2)
check("BPE roundtrip", decoded2 == text2)

print("\n--- Special tokens ---")

tok3 = rp.Tokenizer()
tok3.train("hello world hello world", vocab_size=260, special_tokens=["<PAD>", "<EOS>", "<UNK>"])
check("vocab includes specials", tok3.vocab_size >= 259)

eos_id = tok3.token_to_id("<EOS>")
check("special token has id", eos_id >= 0)

eos_token = tok3.id_to_token(eos_id)
check("id_to_token for special", eos_token == "<EOS>")

pad_id = tok3.token_to_id("<PAD>")
unk_id = tok3.token_to_id("<UNK>")
check("all specials have unique ids", len(set([pad_id, eos_id, unk_id])) == 3)

print("\n--- Special tokens in text ---")

tok4 = rp.Tokenizer()
tok4.train("hello<EOS>world<EOS>hello<EOS>world", vocab_size=260, special_tokens=["<EOS>"])
ids4 = tok4.encode("hello<EOS>world")
check("special token encoded", tok4.token_to_id("<EOS>") in ids4)
decoded4 = tok4.decode(ids4)
check("special token roundtrip", decoded4 == "hello<EOS>world")

print("\n--- Token lookup ---")

tok5 = rp.Tokenizer()
tok5.train("abcabc", vocab_size=260)

a_id = tok5.token_to_id("a")
check("single char lookup", a_id >= 0)
check("id_to_token inverse", tok5.id_to_token(a_id) == "a")

try:
    tok5.token_to_id("nonexistent_xyz")
    check("nonexistent token raises", False)
except KeyError:
    check("nonexistent token raises", True)

try:
    tok5.id_to_token(999999)
    check("invalid id raises", False)
except IndexError:
    check("invalid id raises", True)

print("\n--- Save / Load ---")

tok6 = rp.Tokenizer()
tok6.train("hello world foo bar baz " * 20, vocab_size=280, special_tokens=["<START>", "<END>"])
ids_before = tok6.encode("hello world")
vocab_before = tok6.vocab_size

test_path = "/tmp/test_tokenizer.rptok"
tok6.save(test_path)
check("save succeeds", os.path.exists(test_path))

tok7 = rp.load_tokenizer(test_path)
check("load succeeds", tok7 is not None)
check("loaded vocab_size matches", tok7.vocab_size == vocab_before)

ids_after = tok7.encode("hello world")
check("loaded encode matches", ids_before == ids_after)

decoded_after = tok7.decode(ids_after)
check("loaded decode matches", decoded_after == "hello world")

special_id = tok7.token_to_id("<START>")
check("loaded special tokens", special_id >= 0)

os.remove(test_path)

print("\n--- Empty / edge cases ---")

tok8 = rp.Tokenizer()
tok8.train("a", vocab_size=257)
ids8 = tok8.encode("a")
check("single char text", len(ids8) == 1)
check("single char roundtrip", tok8.decode(ids8) == "a")

ids_empty = tok8.encode("")
check("empty string encode", len(ids_empty) == 0)
check("empty string decode", tok8.decode([]) == "")

print("\n--- Unicode / binary ---")

tok9 = rp.Tokenizer()
tok9.train("café résumé naïve", vocab_size=270)
ids9 = tok9.encode("café")
decoded9 = tok9.decode(ids9)
check("unicode roundtrip", decoded9 == "café")

print("\n--- Large vocab training ---")

large_text = "the quick brown fox jumps over the lazy dog " * 100
tok10 = rp.Tokenizer()
tok10.train(large_text, vocab_size=400)
check("large vocab size", tok10.vocab_size <= 400)
ids10 = tok10.encode("the quick brown fox")
decoded10 = tok10.decode(ids10)
check("large vocab roundtrip", decoded10 == "the quick brown fox")
check(f"large vocab compression ({len(ids10)} tokens for 19 chars)", len(ids10) < 19)

print("\n--- Multiple special tokens in sequence ---")

tok11 = rp.Tokenizer()
tok11.train("hello<SEP>world<SEP>foo<SEP>bar", vocab_size=260, special_tokens=["<SEP>"])
ids11 = tok11.encode("<SEP><SEP><SEP>")
sep_id = tok11.token_to_id("<SEP>")
check("consecutive specials", ids11 == [sep_id, sep_id, sep_id])

print("\n" + "=" * 60)
print(f"Results: {passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("All tokenizer tests passed!")
