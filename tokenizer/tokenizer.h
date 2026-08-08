#ifndef RECPULSE_TOKENIZER_H
#define RECPULSE_TOKENIZER_H

#include <stddef.h>

typedef struct {
    int first;
    int second;
} MergePair;

typedef struct {
    char** vocab;
    int vocab_size;
    int vocab_capacity;

    MergePair* merges;
    int* merge_results;
    int num_merges;

    char** special_tokens;
    int num_special;

    int base_vocab_size;
} Tokenizer;

Tokenizer* tokenizer_create(void);
void tokenizer_free(Tokenizer* tok);

int tokenizer_train(Tokenizer* tok, const char* text, size_t text_len,
                    int target_vocab_size,
                    const char** special_tokens, int num_special);

int* tokenizer_encode(Tokenizer* tok, const char* text, size_t text_len, int* out_len);
char* tokenizer_decode(Tokenizer* tok, const int* ids, int num_ids);

int tokenizer_save(Tokenizer* tok, const char* path);
Tokenizer* tokenizer_load(const char* path);

int tokenizer_vocab_size(Tokenizer* tok);
const char* tokenizer_id_to_token(Tokenizer* tok, int id);
int tokenizer_token_to_id(Tokenizer* tok, const char* token);

#endif
