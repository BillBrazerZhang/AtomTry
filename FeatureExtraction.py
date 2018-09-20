# %% import packages
import numpy
import urllib.request
import scipy.optimize
import random
import nltk
import string
from nltk.stem.porter import *
from sklearn import linear_model
from nltk.corpus import stopwords
try:
    from collections.abc import Mapping, defaultdict
except ImportError:
    from collections import Mapping, defaultdict
    
# %% basic class
class KeywordCount:
    def __init__(self, fname):
        self.name = fname  #file name of input transcript
        #init for updateWordCount()
        self.words = []  #sequenced word list
        self.wordCount = defaultdict(int)  #dictionary of words and their count
        self.wordCountSort = defaultdict(int)  #sorted dictionary
        self.transSize = 0  #words the transcript have
        self.transSizeNS = 0  #meaningful words the transcript have
        self.wordsetSize = 0  #all different words
        #init for updateKeywordStatistics()
        self.keywordStat = defaultdict(dict)  #num(int) certain keyword appears in the transcript
        self.keywordStatSorted = defaultdict(dict)
        self.keywordLoc = defaultdict(dict)  #locations(list) certain keyword appears in the transcript
        self.keywordPart = defaultdict(dict)  #percentage per keyword from all keywords
        self.keywordPer = float(0)  #percentage all keywords from all words
        #init for questionMark()
        self.questionStat = 0
        self.questionLoc = []
        self.questionPer = float(0)
        #init for emoVoice()
        self.emoTerm = []
        self.emoLoc = []
        #init for keywordQuestionmark()
        self.keywordQNum = defaultdict(dict)
        self.keywordQPer = defaultdict(dict)
        self.keywordQOutput = defaultdict(dict)
        #init for nonSpeech()
        self.nSpeech = []
        #init for lengthByTurn()
        self.listTurn = []

        print("Reading data...")
        print("Parsing by words...")
        with open(fname, encoding='utf-8', errors='ignore') as f:
            data = f.read()
        tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.data = tokenizer.tokenize(data)
        print("Parsing by sentences...")
        from nltk.tokenize import sent_tokenize
        self.sentNum = 0
        self.sents = []
        with open(fname, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.lower()
                self.sentNum += len(sent_tokenize(line))
                for sent in sent_tokenize(line):
                    self.sents.append(sent)

        print("done")

        print("Initializing keword dictionary...")
        # Dictionary of key-terms for CTS fidelity
        self.keywordCTS = defaultdict(set)

        self.keywordCTS['Agenda'] = {'agenda','priorities','priority','do first','most important','work on','focus on','talk about' \
                                     ,'plan','todo list' ,'focus on first','focus on today','focus on during the session' \
                                     ,'talk about today','talk about first','talk about during the session' \
                                     ,'work on today','work on first','work on during the session' \
                                     ,'you like to','you want to','add to the agenda','add anything to the agenda','last week' \
                                     ,'evidence','mistake in thinking','what did you think','what did you want' \
                                     ,'how did you feel'}

        self.keywordCTS['Feedback'] = {'feedback','reaction','advise','advice','suggestion','previous','last time','last week','last session','past session' \
                                       ,'think about today','things go today','think about today\'s session','concern' \
                                       ,'what question','what questions','unhelpful','helpful','least helpful','about today\'s session' \
                                       ,'anything i can do better','anything we can do better','concerns about today\'s session','helpful about the session' \
                                       ,'can help you','we can try','learn','take in','skill','learn skills','achieve','goals','goal' \
                                       ,'if i understand you correctly','are you saying','do i have it right','work on your goals' \
                                       ,'was this helpful','how am i doing today','am i explaining things clearly','do you understand' \
                                       ,'do you follow me'}

        self.keywordCTS['Understanding'] = {'understand','understanding','recognize','observe','grasp','comprehend','know','understand why' \
                                            ,'sounds like','you are saying','you were feeling','you felt','i felt','i was feeling' \
                                            ,'see','makes sense','i see','feel that way','feel this way'}

        self.keywordCTS['Interpersonal Effectiveness'] = {'sorry','hard','difficult','tough' \
                                                          ,'dissappointing','stressful','stressed' \
                                                          ,'scary','frightening','upset','upsetting'\
                                                          ,'unfortunate'}

        self.keywordCTS['Collaboration'] = {'choice', 'you want to do','good idea','because','will','help you get your goal'}

        self.keywordCTS['Guided Discovery'] = {'meaning','mean','self','how','why','evidence' \
                                               ,'conclusion','conclude','decide','decision','decided' \
                                               ,'know','proof','tell me more','assume','assumption' \
                                               ,'hypothesis','disprove','facts','fact','solutions' \
                                               ,'brainstorm','solve','alternative','other explanations' \
                                               ,'another way','other way','to think about','to explain','reason'}

        self.keywordCTS['Focus on Key Cognitions'] = {'thinking','tell yourself','through your mind' \
                                                      ,'what did you tell yourself','what went through your mind' \
                                                      ,'thought','think','connection','lead to','connected' \
                                                      ,'connect','link','linked','make you','you do','feel about the thought'}

        self.keywordCTS['Choices of Intervention'] = {}

        self.keywordCTS['Homework'] = {'homework','review','homework review','at home','practice','assignment','assign','get in the way of','work around' \
                                       ,'assigned','progress','learned','improve','learn','skills','skill','out of session','outside of session' \
                                       ,'goal','better','barrier','in the way','expect','problems','problem','succeed','success'}

        self.keywordCTS['Social Skills Training'] = {'rational','help you learn this skill','help you with your goal' \
                                                     ,'demonstrate','to make your next role','play better','play even better' \
                                                     ,'try to focus on','do well','did well','did a good job' \
                                                     ,'for the next role play','recommend focusing on'}


    def updateWordCount(self):
        print('Starting counting words...')
        stopWords = set(stopwords.words("english"))
        # Ignore capitalization and remove punctuation
        punctuation = set(string.punctuation)
        stemmer = PorterStemmer()
        for d in self.data:
            r = ''.join([c for c in d if not c in punctuation])
            if r != '':
                w = stemmer.stem(r.lower())
                self.words.append(w)
                self.transSize += 1
                if not w in stopWords:
                    self.wordCount[w] += 1
                    self.transSizeNS += 1
        self.wordsetSize = len(self.wordCount)
        self.wordCountSort = sorted(self.wordCount.items(), key = lambda kv: kv[1])
        print('Word counting done.')

    def updateKeywordStatistics(self):
        print('Starting keyword statisics...')
        s = 0
        for k1 in self.keywordCTS.keys():
            for k2 in self.keywordCTS[k1]:
                self.keywordStat[k1][k2] = 0
                if len(k2.split()) == 1:
                    if k2 in self.wordCount.keys():
                        self.keywordStat[k1][k2] = self.wordCount[k2]
                        s += self.wordCount[k2]
                        loc = []
                        cap = len(self.words)
                        for i in range(cap):
                            if self.words[i] == k2:
                                loc.append(float(i)/float(cap))
                        self.keywordLoc[k1][k2] = loc
                else:
                    loc = []
                    for i in range(self.sentNum):
                        if k2 in self.sents[i]:
                            self.keywordStat[k1][k2] += self.sents[i].count(k2)
                            s += self.sents[i].count(k2)
                            for j in range(self.sents[i].count(k2)):
                                loc.append(float(i)/float(self.sentNum))
                    if self.keywordStat[k1][k2] > 0:
                        self.keywordLoc[k1][k2] = loc
        for k1 in self.keywordStat.keys():
          self.keywordStatSorted[k1] = sorted(self.keywordStat[k1].items(), key = lambda kv: kv[1])
        for k1 in self.keywordStat.keys():
            for k2 in self.keywordStat[k1].keys():
                self.keywordPart[k1][k2] = self.keywordStat[k1][k2]/float(s)
        self.keywordPer = s/float(self.transSize)
        print('Keyword statistics done.')

    def questionMark(self):
        print('Starting questionmark statistics...')
        l = len(self.data)
        for i in range(l):
            if self.data[i] == '?':
                self.questionStat += 1
                self.questionLoc.append(float(i)/float(l))
        self.questionPer = float(self.questionStat)/float(self.sentNum)
        print('Questionmark statistics done.')

    def emoVoice(self):
        i = 0
        with open(self.name, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.lower()
                if not (line[0] == 'p' or line[0] == 'c'):
                    self.emoTerm.append(line)
                    self.emoLoc.append((line,float(i)/float(self.sentNum)))
                i += 1

    def keywordQuestionmark(self):
        print('Starting keyword in questionmarks statistics...')
        for k1 in self.keywordCTS.keys():
            for k2 in self.keywordCTS[k1]:
                if self.keywordStat[k1][k2] > 0:
                    self.keywordQNum[k1][k2] = 0
                    for s in self.sents:
                        self.keywordQNum[k1][k2] += int('?' in s * s.count(k2))
                    self.keywordQPer[k1][k2] = float(self.keywordQNum[k1][k2])/self.keywordStat[k1][k2]
                    self.keywordQOutput[k1][k2] = (self.keywordQNum[k1][k2],self.keywordStat[k1][k2],self.keywordQPer[k1][k2])
        print('keyword with questionmarks statistics done.')

    def nonSpeech(self):
        with open(self.name, encoding='utf-8', errors='ignore') as f:
            data = f.read()
        flag = False
        start = []
        end = []
        for i in range(len(data)):
            if data[i] == '[':
                start.append(i)
                j = i
                flag = True
                while(flag == True and j < len(data)-1):
                    j += 1
                    if data[j] == ']' or j == len(data)-1:
                        break
                if data[j] == ']':
                    end.append(j)
        for k in range(len(end)):
            self.nSpeech.append(data[start[k]:end[k]+1])

    def lengthByTurn(self):
        tokenizer = nltk.tokenize.TreebankWordTokenizer()
        with open(self.name, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.lower()
                if (line[0] == 'c'):
                    self.listTurn.append(('c',len(tokenizer.tokenize(line))))
                elif (line[0] == 'p'):
                    self.listTurn.append(('p',len(tokenizer.tokenize(line))))



# %% child class
class LabelledKeywordCount(KeywordCount):
    def __init__(self, fname, label):

        self.name = fname  #file name of input transcript
        self.label = label
        #init for updateWordCount()
        self.words = []  #sequenced word list
        self.wordCount = defaultdict(int)  #dictionary of words and their count
        self.wordCountSort = defaultdict(int)  #sorted dictionary
        self.transSize = 0  #words the transcript have
        self.transSizeNS = 0  #meaningful words the transcript have
        self.wordsetSize = 0  #all different words
        #init for updateKeywordStatistics()
        self.keywordStat = defaultdict(dict)  #num(int) certain keyword appears in the transcript
        self.keywordStatSorted = defaultdict(dict)
        self.keywordLoc = defaultdict(dict)  #locations(list) certain keyword appears in the transcript
        self.keywordPart = defaultdict(dict)  #percentage per keyword from all keywords
        self.keywordPer = float(0)  #percentage all keywords from all words
        #init for questionMark()
        self.questionStat = 0
        self.questionLoc = []
        self.questionPer = float(0)
        #init for emoVoice()
        self.emoTerm = []
        self.emoLoc = []
        #init for keywordQuestionmark()
        self.keywordQNum = defaultdict(dict)
        self.keywordQPer = defaultdict(dict)
        self.keywordQOutput = defaultdict(dict)
        #init for nonSpeech()
        self.nSpeech = []
        #init for lengthByTurn()
        self.listTurn = []

        print("Reading data...")
        from nltk.tokenize import sent_tokenize
        self.data = []
        self.lines = []
        self.sents = []
        self.sentNum = 0
        with open(fname, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.lower()
                if (line[0] == label):
                    self.lines.append(sent_tokenize(line))
                    for sent in sent_tokenize(line):
                        self.sents.append(sent)
                    self.sentNum += len(sent_tokenize(line))
                    tokenizer = nltk.tokenize.TreebankWordTokenizer()
                    self.data += tokenizer.tokenize(line)
        print("done")

        print("Initializing keyword dictionary...")
        # Dictionary of key-terms for CTS fidelity
        self.keywordCTS = defaultdict(set)

        self.keywordCTS['Agenda'] = {'agenda','priorities','priority','do first','most important','work on','focus on','talk about' \
                                     ,'plan','todo list' ,'focus on first','focus on today','focus on during the session' \
                                     ,'talk about today','talk about first','talk about during the session' \
                                     ,'work on today','work on first','work on during the session' \
                                     ,'you like to','you want to','add to the agenda','add anything to the agenda','last week' \
                                     ,'evidence','mistake in thinking','what did you think','what did you want' \
                                     ,'how did you feel'}

        self.keywordCTS['Feedback'] = {'feedback','reaction','advise','advice','suggestion','previous','last time','last week','last session','past session' \
                                       ,'think about today','things go today','think about today\'s session','concern' \
                                       ,'what question','what questions','unhelpful','helpful','least helpful','about today\'s session' \
                                       ,'anything i can do better','anything we can do better','concerns about today\'s session','helpful about the session' \
                                       ,'can help you','we can try','learn','take in','skill','learn skills','achieve','goals','goal' \
                                       ,'if i understand you correctly','are you saying','do i have it right','work on your goals' \
                                       ,'was this helpful','how am i doing today','am i explaining things clearly','do you understand' \
                                       ,'do you follow me'}

        self.keywordCTS['Understanding'] = {'understand','understanding','recognize','observe','grasp','comprehend','know','understand why' \
                                            ,'sounds like','you are saying','you were feeling','you felt','i felt','i was feeling' \
                                            ,'see','makes sense','i see','feel that way','feel this way'}

        self.keywordCTS['Interpersonal Effectiveness'] = {'sorry','hard','difficult','tough' \
                                                          ,'dissappointing','stressful','stressed' \
                                                          ,'scary','frightening','upset','upsetting'\
                                                          ,'unfortunate'}

        self.keywordCTS['Collaboration'] = {'choice', 'you want to do','good idea','because','will','help you get your goal'}

        self.keywordCTS['Guided Discovery'] = {'meaning','mean','self','how','why','evidence' \
                                               ,'conclusion','conclude','decide','decision','decided' \
                                               ,'know','proof','tell me more','assume','assumption' \
                                               ,'hypothesis','disprove','facts','fact','solutions' \
                                               ,'brainstorm','solve','alternative','other explanations' \
                                               ,'another way','other way','to think about','to explain','reason'}

        self.keywordCTS['Focus on Key Cognitions'] = {'thinking','tell yourself','through your mind' \
                                                      ,'what did you tell yourself','what went through your mind' \
                                                      ,'thought','think','connection','lead to','connected' \
                                                      ,'connect','link','linked','make you','you do','feel about the thought'}

        self.keywordCTS['Choices of Intervention'] = {}

        self.keywordCTS['Homework'] = {'homework','review','homework review','at home','practice','assignment','assign','get in the way of','work around' \
                                       ,'assigned','progress','learned','improve','learn','skills','skill','out of session','outside of session' \
                                       ,'goal','better','barrier','in the way','expect','problems','problem','succeed','success'}

        self.keywordCTS['Social Skills Training'] = {'rational','help you learn this skill','help you with your goal' \
                                                     ,'demonstrate','to make your next role','play better','play even better' \
                                                     ,'try to focus on','do well','did well','did a good job' \
                                                     ,'for the next role play','recommend focusing on'}
