import editdistance
import random
import numpy as np
class TextGenerator(object):
    def __init__(self,ratios=[1,1,1,5],chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
        self.func = []
        for ratio, func in zip(ratios,[self._insert,self._delete,self._change,self._keep]):
            self.func.extend([func]*ratio)
        self.chars = chars
    def __call__(self,word):
        idx = 0
        new_word = word
        while idx < len(new_word):
            new_word = random.choice(self.func)(new_word,idx)
            idx +=1
        return new_word
    def _insert(self,word, idx):
        assert idx < len(word)
        return word[:idx]+random.choice(self.chars)+word[idx:]
    def _delete(self,word, idx):
        assert idx < len(word)
        return word[:idx]+word[idx+1:]
    def _change(self,word, idx):
        assert idx < len(word)
        return word[:idx]+random.choice(self.chars)+word[idx+1:]
    def _keep(self,word, idx):
        return word
    def calculate_similarity_matric(self, words1, words2):
        similarity = np.zeros([len(words1), len(words2)])
        for i,word1 in enumerate(words1):
            for j,word2 in enumerate(words2):
                similarity[i,j] = 1-editdistance.eval(word1,word2)/max(len(word1), len(word2))
        return similarity
    def filter_words(self, texts):
        idxs,new_texts = [], []
        for idx, text in enumerate(texts):
            text = text.lower()
            char_list = [c for c in text if c in self.chars]
            if len(char_list)==0:
                continue
            idxs.append(idx)
            new_texts.append("".join(char_list))
        return idxs, new_texts

        
# generator = TextGenerator()
# words1 = ["hello","how","morning"]
# words2 = ["hello9999999","howe","mofhvfing","nothing"]
# sim = generator.calculate_similarity_matric(words1,words2)
# print(sim.shape)
# print(sim)
# sim = generator.calculate_similarity_matric(words2,words1)
# print(sim)
# print(editdistance.eval('hello','heooo'))
# print(generator("word"))