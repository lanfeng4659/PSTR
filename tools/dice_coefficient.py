import editdistance
import argparse
import time
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', default='crop')
parser.add_argument('-b', default='crop')
def dice_coefficient(a,b):
    if not len(a) or not len(b): return 0.0
    """ quick case for true duplicates """
    if a == b: return 1.0
    """ if a != b, and a or b are single chars, then they can't possibly match """
    if len(a) == 1 or len(b) == 1: return 0.0
    
    """ use python list comprehension, preferred over list.append() """
    a_bigram_list = [a[i:i+2] for i in range(len(a)-1)]
    b_bigram_list = [b[i:i+2] for i in range(len(b)-1)]
    
    a_bigram_list.sort()
    b_bigram_list.sort()
    
    # assignments to save function calls
    lena = len(a_bigram_list)
    lenb = len(b_bigram_list)
    # initialize match counters
    matches = i = j = 0
    while (i < lena and j < lenb):
        if a_bigram_list[i] == b_bigram_list[j]:
            matches += 2
            i += 1
            j += 1
        elif a_bigram_list[i] < b_bigram_list[j]:
            i += 1
        else:
            j += 1
    print(lena, lenb, matches)
    score = float(2*matches)/float(lena + lenb)
    return score
from strsimpy.cosine import Cosine
cosine = Cosine(2)
def cossim(a,b):
    p0 = cosine.get_profile(a)
    p1 = cosine.get_profile(b)
    score = cosine.similarity_profiles(p0, p1)
    print(p0)
    print(p1)
    print(score, len(p0), len(p1))
    return score

if __name__ == '__main__':
    args = parser.parse_args()
    print(cossim(args.a,args.b))