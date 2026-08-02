#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static char* strdup_safe(const char* s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    char* d = (char*)malloc(len + 1);
    if (d) memcpy(d, s, len + 1);
    return d;
}

static int vocab_add(Tokenizer* tok, const char* token) {
    if (tok->vocab_size >= tok->vocab_capacity) {
        int new_cap = tok->vocab_capacity * 2;
        char** new_vocab = (char**)realloc(tok->vocab, new_cap * sizeof(char*));
        if (!new_vocab) return -1;
        tok->vocab = new_vocab;
        tok->vocab_capacity = new_cap;
    }
    tok->vocab[tok->vocab_size] = strdup_safe(token);
    return tok->vocab_size++;
}

static char* concat_tokens(const char* a, const char* b) {
    size_t la = strlen(a);
    size_t lb = strlen(b);
    char* result = (char*)malloc(la + lb + 1);
    if (!result) return NULL;
    memcpy(result, a, la);
    memcpy(result + la, b, lb);
    result[la + lb] = '\0';
    return result;
}

Tokenizer* tokenizer_create(void) {
    Tokenizer* tok = (Tokenizer*)calloc(1, sizeof(Tokenizer));
    if (!tok) return NULL;
    tok->vocab_capacity = 512;
    tok->vocab = (char**)calloc(tok->vocab_capacity, sizeof(char*));
    if (!tok->vocab) { free(tok); return NULL; }
    return tok;
}

void tokenizer_free(Tokenizer* tok) {
    if (!tok) return;
    for (int i = 0; i < tok->vocab_size; i++) free(tok->vocab[i]);
    free(tok->vocab);
    free(tok->merges);
    free(tok->merge_results);
    if (tok->special_tokens) {
        for (int i = 0; i < tok->num_special; i++) free(tok->special_tokens[i]);
        free(tok->special_tokens);
    }
    free(tok);
}

static int find_special_token(Tokenizer* tok, const char* text, size_t pos, size_t text_len, int* token_id, size_t* match_len) {
    for (int i = 0; i < tok->num_special; i++) {
        size_t slen = strlen(tok->special_tokens[i]);
        if (pos + slen <= text_len && memcmp(text + pos, tok->special_tokens[i], slen) == 0) {
            for (int j = 0; j < tok->vocab_size; j++) {
                if (strcmp(tok->vocab[j], tok->special_tokens[i]) == 0) {
                    *token_id = j;
                    *match_len = slen;
                    return 1;
                }
            }
        }
    }
    return 0;
}

typedef struct {
    int first;
    int second;
    int count;
} PairCount;

static int find_most_frequent_pair(const int* tokens, int num_tokens, int num_special, int* best_first, int* best_second) {
    if (num_tokens < 2) return 0;

    int capacity = 1024;
    PairCount* table = (PairCount*)calloc(capacity, sizeof(PairCount));
    if (!table) return 0;
    int table_size = 0;

    for (int i = 0; i < num_tokens - 1; i++) {
        int a = tokens[i], b = tokens[i + 1];
        if (a < num_special || b < num_special) continue;
        unsigned int hash = ((unsigned int)a * 73856093u ^ (unsigned int)b * 19349663u) % (unsigned int)capacity;

        int found = 0;
        for (int j = 0; j < capacity; j++) {
            int idx = (hash + j) % capacity;
            if (table[idx].count == 0) {
                table[idx].first = a;
                table[idx].second = b;
                table[idx].count = 1;
                table_size++;
                found = 1;
                break;
            }
            if (table[idx].first == a && table[idx].second == b) {
                table[idx].count++;
                found = 1;
                break;
            }
        }

        if (!found || table_size > capacity * 3 / 4) {
            int new_cap = capacity * 2;
            PairCount* new_table = (PairCount*)calloc(new_cap, sizeof(PairCount));
            if (!new_table) { free(table); return 0; }
            for (int j = 0; j < capacity; j++) {
                if (table[j].count > 0) {
                    unsigned int h = ((unsigned int)table[j].first * 73856093u ^ (unsigned int)table[j].second * 19349663u) % (unsigned int)new_cap;
                    for (int k = 0; k < new_cap; k++) {
                        int ni = (h + k) % new_cap;
                        if (new_table[ni].count == 0) {
                            new_table[ni] = table[j];
                            break;
                        }
                    }
                }
            }
            free(table);
            table = new_table;
            capacity = new_cap;

            if (!found) {
                unsigned int h2 = ((unsigned int)a * 73856093u ^ (unsigned int)b * 19349663u) % (unsigned int)capacity;
                for (int j2 = 0; j2 < capacity; j2++) {
                    int idx2 = (h2 + j2) % capacity;
                    if (table[idx2].count == 0) {
                        table[idx2].first = a;
                        table[idx2].second = b;
                        table[idx2].count = 1;
                        table_size++;
                        break;
                    }
                    if (table[idx2].first == a && table[idx2].second == b) {
                        table[idx2].count++;
                        break;
                    }
                }
            }
        }
    }

    int best_count = 0;
    *best_first = -1;
    *best_second = -1;
    for (int i = 0; i < capacity; i++) {
        if (table[i].count > best_count) {
            best_count = table[i].count;
            *best_first = table[i].first;
            *best_second = table[i].second;
        }
    }

    free(table);
    return best_count;
}

static int apply_merge(int* tokens, int num_tokens, int first, int second, int new_id, int num_special) {
    int write = 0;
    for (int read = 0; read < num_tokens; read++) {
        if (read < num_tokens - 1 && tokens[read] == first && tokens[read + 1] == second
            && first >= num_special && second >= num_special) {
            tokens[write++] = new_id;
            read++;
        } else {
            tokens[write++] = tokens[read];
        }
    }
    return write;
}

int tokenizer_train(Tokenizer* tok, const char* text, size_t text_len,
                    int target_vocab_size,
                    const char** special_tokens, int num_special) {
    if (!tok || !text || target_vocab_size < 256) return -1;

    tok->num_special = num_special;
    if (num_special > 0) {
        tok->special_tokens = (char**)malloc(num_special * sizeof(char*));
        for (int i = 0; i < num_special; i++) {
            tok->special_tokens[i] = strdup_safe(special_tokens[i]);
        }
    }

    for (int i = 0; i < num_special; i++) {
        vocab_add(tok, special_tokens[i]);
    }

    for (int i = 0; i < 256; i++) {
        char byte_str[2] = {(char)i, '\0'};
        vocab_add(tok, byte_str);
    }

    tok->base_vocab_size = tok->vocab_size;

    int* tokens = (int*)malloc(text_len * sizeof(int));
    if (!tokens) return -1;
    int num_tokens = 0;

    size_t pos = 0;
    while (pos < text_len) {
        int special_id;
        size_t match_len;
        if (find_special_token(tok, text, pos, text_len, &special_id, &match_len)) {
            tokens[num_tokens++] = special_id;
            pos += match_len;
        } else {
            unsigned char byte = (unsigned char)text[pos];
            tokens[num_tokens++] = tok->num_special + (int)byte;
            pos++;
        }
    }

    int max_merges = target_vocab_size - tok->vocab_size;
    if (max_merges <= 0) { free(tokens); return 0; }

    tok->merges = (MergePair*)malloc(max_merges * sizeof(MergePair));
    tok->merge_results = (int*)malloc(max_merges * sizeof(int));
    tok->num_merges = 0;

    for (int step = 0; step < max_merges; step++) {
        int best_first, best_second;
        int count = find_most_frequent_pair(tokens, num_tokens, tok->num_special, &best_first, &best_second);
        if (count < 2) break;

        char* merged = concat_tokens(tok->vocab[best_first], tok->vocab[best_second]);
        if (!merged) break;

        int new_id = vocab_add(tok, merged);
        free(merged);

        tok->merges[tok->num_merges].first = best_first;
        tok->merges[tok->num_merges].second = best_second;
        tok->merge_results[tok->num_merges] = new_id;
        tok->num_merges++;

        num_tokens = apply_merge(tokens, num_tokens, best_first, best_second, new_id, tok->num_special);
    }

    free(tokens);
    return 0;
}

int* tokenizer_encode(Tokenizer* tok, const char* text, size_t text_len, int* out_len) {
    if (!tok || !text || !out_len) return NULL;

    int* tokens = (int*)malloc((text_len + 1) * sizeof(int));
    if (!tokens) return NULL;
    int num_tokens = 0;

    size_t pos = 0;
    while (pos < text_len) {
        int special_id;
        size_t match_len;
        if (find_special_token(tok, text, pos, text_len, &special_id, &match_len)) {
            tokens[num_tokens++] = special_id;
            pos += match_len;
        } else {
            unsigned char byte = (unsigned char)text[pos];
            tokens[num_tokens++] = tok->num_special + (int)byte;
            pos++;
        }
    }

    for (int m = 0; m < tok->num_merges; m++) {
        num_tokens = apply_merge(tokens, num_tokens,
                                 tok->merges[m].first, tok->merges[m].second,
                                 tok->merge_results[m], tok->num_special);
    }

    *out_len = num_tokens;
    return tokens;
}

char* tokenizer_decode(Tokenizer* tok, const int* ids, int num_ids) {
    if (!tok || !ids) return NULL;

    size_t total_len = 0;
    for (int i = 0; i < num_ids; i++) {
        if (ids[i] >= 0 && ids[i] < tok->vocab_size) {
            total_len += strlen(tok->vocab[ids[i]]);
        }
    }

    char* result = (char*)malloc(total_len + 1);
    if (!result) return NULL;

    size_t pos = 0;
    for (int i = 0; i < num_ids; i++) {
        if (ids[i] >= 0 && ids[i] < tok->vocab_size) {
            size_t len = strlen(tok->vocab[ids[i]]);
            memcpy(result + pos, tok->vocab[ids[i]], len);
            pos += len;
        }
    }
    result[pos] = '\0';

    return result;
}

int tokenizer_save(Tokenizer* tok, const char* path) {
    if (!tok || !path) return -1;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    int header[5] = {0x52505431, tok->vocab_size, tok->num_merges, tok->num_special, tok->base_vocab_size};
    fwrite(header, sizeof(int), 5, f);

    for (int i = 0; i < tok->num_special; i++) {
        int slen = (int)strlen(tok->special_tokens[i]);
        fwrite(&slen, sizeof(int), 1, f);
        fwrite(tok->special_tokens[i], 1, slen, f);
    }

    for (int i = 0; i < tok->vocab_size; i++) {
        int vlen = (int)strlen(tok->vocab[i]);
        fwrite(&vlen, sizeof(int), 1, f);
        fwrite(tok->vocab[i], 1, vlen, f);
    }

    for (int i = 0; i < tok->num_merges; i++) {
        int merge_data[3] = {tok->merges[i].first, tok->merges[i].second, tok->merge_results[i]};
        fwrite(merge_data, sizeof(int), 3, f);
    }

    fclose(f);
    return 0;
}

Tokenizer* tokenizer_load(const char* path) {
    if (!path) return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    int header[5];
    if (fread(header, sizeof(int), 5, f) != 5 || header[0] != 0x52505431) {
        fclose(f);
        return NULL;
    }

    int vocab_size = header[1];
    int num_merges = header[2];
    int num_special = header[3];
    int base_vocab_size = header[4];

    Tokenizer* tok = tokenizer_create();
    if (!tok) { fclose(f); return NULL; }

    tok->num_special = num_special;
    if (num_special > 0) {
        tok->special_tokens = (char**)malloc(num_special * sizeof(char*));
        for (int i = 0; i < num_special; i++) {
            int slen;
            if (fread(&slen, sizeof(int), 1, f) != 1) { tokenizer_free(tok); fclose(f); return NULL; }
            char* s = (char*)malloc(slen + 1);
            if ((int)fread(s, 1, slen, f) != slen) { free(s); tokenizer_free(tok); fclose(f); return NULL; }
            s[slen] = '\0';
            tok->special_tokens[i] = s;
        }
    }

    for (int i = 0; i < vocab_size; i++) {
        int vlen;
        if (fread(&vlen, sizeof(int), 1, f) != 1) { tokenizer_free(tok); fclose(f); return NULL; }
        char* token = (char*)malloc(vlen + 1);
        if ((int)fread(token, 1, vlen, f) != vlen) { free(token); tokenizer_free(tok); fclose(f); return NULL; }
        token[vlen] = '\0';
        vocab_add(tok, token);
        free(token);
    }

    tok->base_vocab_size = base_vocab_size;
    tok->num_merges = num_merges;
    tok->merges = (MergePair*)malloc(num_merges * sizeof(MergePair));
    tok->merge_results = (int*)malloc(num_merges * sizeof(int));

    for (int i = 0; i < num_merges; i++) {
        int merge_data[3];
        if (fread(merge_data, sizeof(int), 3, f) != 3) { tokenizer_free(tok); fclose(f); return NULL; }
        tok->merges[i].first = merge_data[0];
        tok->merges[i].second = merge_data[1];
        tok->merge_results[i] = merge_data[2];
    }

    fclose(f);
    return tok;
}

int tokenizer_vocab_size(Tokenizer* tok) {
    return tok ? tok->vocab_size : 0;
}

const char* tokenizer_id_to_token(Tokenizer* tok, int id) {
    if (!tok || id < 0 || id >= tok->vocab_size) return NULL;
    return tok->vocab[id];
}

int tokenizer_token_to_id(Tokenizer* tok, const char* token) {
    if (!tok || !token) return -1;
    for (int i = 0; i < tok->vocab_size; i++) {
        if (strcmp(tok->vocab[i], token) == 0) return i;
    }
    return -1;
}
