import editdistance
import random
import numpy as np
import torch
import thulac
# from multiprocessing.dummy import Pool as ThreadPool
from strsimpy.cosine import Cosine
def nor_editdistance(word1, word2):
    return 1-editdistance.eval(word1,word2)/max(len(word1), len(word2))
cosine = Cosine(2)
def cossim(a, b):
    if not len(a) or not len(b): return 0.0
    """ quick case for true duplicates """
    if a == b: return 1.0
    """ if a != b, and a or b are single chars, then they can't possibly match """
    if len(a) == 1 or len(b) == 1: return 0.0

    p0 = cosine.get_profile(a)
    p1 = cosine.get_profile(b)
    score = cosine.similarity_profiles(p0, p1)
    return score
global cutter
cutter = thulac.thulac(seg_only=True)
class TextGenerator(object):
    def __init__(self,ratios=[1,1,1,5],chars='abcdefghijklmnopqrstuvwxyz0123456789',sim_fn='nor_editdistance'):
        self.func = []
        for ratio, func in zip(ratios,[self._insert,self._delete,self._change,self._keep]):
            self.func.extend([func]*ratio)
        self.chars = chars
        self.char_to_label_map = {}
        self.is_chinese = len(self.chars)>1000
        # self.cutter = thulac.thulac(seg_only=True)
        for i, c in enumerate(self.chars):
            self.char_to_label_map[c] = i
        assert sim_fn in ["nor_editdistance", "cossim"]
        self.sim_fn = eval(sim_fn)
    def __call__(self,word):
        idx = 0
        new_word = word
        while idx < len(new_word):
            new_word = random.choice(self.func)(new_word,idx)
            idx +=1
        return new_word
    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        alphabet = self.chars + ['-']
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([alphabet[i] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != len(alphabet)-1 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(alphabet[t[i]])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
    def label_map(self, word):
        if self.is_chinese:
            result = [self.char_to_label_map[char] for char in word if char in self.chars]
            # print(word,result)
            return result
        else:
            return [self.char_to_label_map[char] for char in word if char.lower() in self.chars]
    def label_map_with_padding(self, word, max_len=15, padding=0):
        labels = [padding]*max_len
        if self.is_chinese:
            for i,char in enumerate(word):
                if i >= max_len:
                    continue
                if char in self.chars:
                    labels[i] = self.char_to_label_map[char]
                # result = [self.char_to_label_map[char] for char in word if char in self.chars]
            # print(word,result)
            return labels
        else:
            for i,char in enumerate(word):
                if i >= max_len:
                    continue
                char = char.lower()
                if char in self.chars:
                    labels[i] = self.char_to_label_map[char]
            return labels
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
    def similarity_on_pair(self,a,b,dis=0):
        a_len = len(a)
        b_len = len(b)
        op = [[0]*(b_len+1) for i in range(a_len+1)]
        si = [[0]*(b_len+1) for i in range(a_len+1)]
        for i in range(b_len+1):
            op[0][i] = i
            si[0][i] = 0
        for i in range(a_len+1):
            op[i][0] = i
            si[i][0] = 0
        for i in range(1, a_len+1):
            for j in range(1, b_len+1):
                if a[i-1]==b[j-1]:
                    op[i][j] = op[i-1][j-1]
                else:
                    op[i][j] = min([op[i-1][j], op[i][j-1], op[i-1][j-1]])+1
                si[i][j] = 1-op[i][j]/max([i,j])
        # print(si)
        return si
    # def resize_diag(self,si,max_len):
    #     map2 = torch.nn.functional.interpolate(si[None,None,...], size=(max_len,max_len),mode='bilinear', align_corners=True)[0,0,...]
    #     return torch.diagonal(map2)
    def resize_diag(self,si,max_len):
        # return torch.zeros([15])
        map2 = torch.nn.functional.interpolate(si[None,None,...], size=(max_len,max_len),mode='bilinear', align_corners=True)[0,0,...]
        return torch.diagonal(map2)
    def editdistance(self, word1, word2):
        return 1-editdistance.eval(word1,word2)/max(len(word1), len(word2))
    def calculate_similarity_matric(self, words1, words2):
        similarity = np.zeros([len(words1), len(words2)])
        for i,word1 in enumerate(words1):
            for j,word2 in enumerate(words2):
                # similarity[i,j] = self.editdistance(word1, word2)
                similarity[i,j] = self.sim_fn(word1, word2)
        return similarity
    def phoc_level_1(self, words):
        phoc1 = np.array([[1 if c in word else 0 for c in self.chars] for word in words]).reshape([len(words),len(self.chars)])
        return phoc1
    def calculate_along_similarity_matric(self,similarity, words1, words2,use_self=False):
        # device = similarity.device
        max_len = similarity.size(2)
        for i,word1 in enumerate(words1):
            for j,word2 in enumerate(words2):
                if use_self:
                    temp = self.resize_diag(torch.tensor(self.similarity_on_pair(word1, word2))[1:,1:],max_len)
                    print(temp)
                    # self.similarity_on_pair(word1, word2)
                else:
                    # torch.tensor is time consuming
                    # similarity[i,j,:] = self.resize_diag(torch.tensor(gen_geo_map.similarity_on_pair(np.array([word1]), np.array([word2]))),max_len)
                    temp = torch.tensor(editdistance.alsm(word1,word2))
                    print(temp)
                    # editdistance.alsm(word1,word2)
        return similarity
    def compare_string(self, words1, words2):
        similarity = np.zeros([len(words1), len(words2)])
        for i,word1 in enumerate(words1):
            for j,word2 in enumerate(words2):
                similarity[i,j] = 1 if word1==word2 else 0
        return similarity
    def filter_words(self, texts):
        idxs,new_texts = [], []
        for idx, text in enumerate(texts):
            text = text.lower()
            char_list = [c for c in text if c in self.chars]
            # print(self.chars)
            if len(char_list)<2:
                continue
            idxs.append(idx)
            new_texts.append("".join(char_list))
        return idxs, new_texts
    def cut_words(self, texts):
        new_words = [v for v in texts]
        for word in texts:
            segs = cutter.cut(word, text=True)
            for seg in segs.split(" "):
                new_words.append(seg)
        return new_words
