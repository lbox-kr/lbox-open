from rouge_score.tokenizers import Tokenizer


class WhiteSpaceTokenizer(Tokenizer):
    def tokenize(self, text):
        return text.split()
