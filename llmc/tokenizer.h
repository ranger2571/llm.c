/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/

#include <stdint.h>
#include <ctype.h>
#include <assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "utils.h"

// ----------------------------------------------------------------------------

typedef struct {
    uint32_t vocab_size;
    char **token_table;
    int init_ok;
    int eot_token; // <|endoftext|> token id
} Tokenizer;

void safe_printf(const char *piece) {
    //因为不能确定piece的值是空或空字符或换行符，所以需要增加判断语句来保证safe print
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

void tokenizer_init(Tokenizer *tokenizer, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        // try to be more helpful as we just added this feature, erase later
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        return;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, sizeof(uint32_t), 256, file);
    assert(header[0] == 20240328);
    int version = header[1];
    tokenizer->vocab_size = header[2];
    if (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        assert(tokenizer->vocab_size == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } else if (version == 2) {
        tokenizer->eot_token = header[3];
    } else {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    // read in all the tokens
    unsigned char length;
    tokenizer->token_table = (char **)mallocCheck(tokenizer->vocab_size * sizeof(char *));
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        /*
        fread用于从文件流中读取数据，其原型是：size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
        它的作用是读取nmemb个元素，每个元素大小为size字节，将它们存储在ptr指向的内存中，并返回成功读取的元素个数。
        如果返回值小于nmemb，可能是因为到达文件末尾（EOF）或发生了错误。

        fread_check函数，其参数包括ptr、size、nmemb、stream，以及文件名和行号。
        函数内部调用了fread，并检查返回的result是否等于预期的nmemb。如果不等于，则处理错误情况。
        这里的关键点在于，这个函数不仅检查了读取的元素数量，还通过feof和ferror来区分不同的错误类型，如文件结束、读取错误或部分读取。当检测到错误时，函数会输出详细的错误信息，包括发生错误的文件和行号，并终止程序。
        */
        freadCheck(&length, sizeof(unsigned char), 1, file);
        //这一行的作用就是从file中读取一个unsigned char类型的数据并存到length中
        assert(length > 0); // every token should be at least one character 每一次存到length中，length长度都应该大于0
        char *token_bytes = (char *)mallocCheck(length + 1);// 将length赋给tokenbytes
        freadCheck(token_bytes, sizeof(char), length, file);
        token_bytes[length] = '\0';  // 增加终止符 Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;//
    }
    //所以整个函数的作用就是逐个的抽取文件中的vocabsize的token，放到tokentable中
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    if (tokenizer->init_ok == 0) {//确定是否完成初始化
        return NULL;
    }
    if (token_id < tokenizer->vocab_size) {//确定是否在vocabsize内
        return tokenizer->token_table[token_id];
    } else {
        printf("invalid token id %u!\n", token_id);
        return NULL;
    }
}

void tokenizer_free(Tokenizer *tokenizer) {
    if (tokenizer->init_ok) {
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->token_table[i]);//free没什么好说的
        }
        free(tokenizer->token_table);
    }
}
