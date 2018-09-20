from FeatureExtraction import KeywordCount
from FeatureExtraction import LabelledKeywordCount

def outputGlobal(fname):
    t = KeywordCount(fname)
    t.updateWordCount()
    t.updateKeywordStatistics()
    t.questionMark()
    t.emoVoice()
    t.keywordQuestionmark()
    t.nonSpeech()
    t.lengthByTurn()

    f = open('FeaturesOutput_'+t.name,'w')
    #file information
    f.write('File name: ' + t.name + '\n')
    f.write('========================================================================================' + '\n')
    f.write('Object: ' + 'Overall' + '\n')
    f.write('========================================================================================' + '\n')
    # f.write('Keyword Dictionary:' + '\n')
    # f.write(str(t.keywordCTS))
    # f.write('\n')
    f.write('========================================================================================' + '\n')
    f.write('========================================================================================' + '\n')
    f.write('========================================================================================' + '\n')
    
    # 1. keywords & key phrases of each domain
    # 1-1. appearing times(sorted)
    f.write('Feature 1-1: times(int) certain keyword appears in the transcript ' + '\n')
    f.write(str(t.keywordStatSorted))
    f.write('\n')
    f.write('========================================================================================' + '\n')
    #1-2. appearing locations(sequenced)
    f.write('Feature 1-2: locations(list of floats) certain keyword appears in the transcrip' + '\n')
    f.write(str(t.keywordLoc))
    f.write('\n')
    f.write('========================================================================================' + '\n')
    # f.write('Feature 3: percentage per keyword from all keywords' + '\n')
    # f.write(str(t.keywordPart))
    # f.write('\n')
    # f.write('========================================================================================' + '\n')

    # 2. keywords & key phrases overall
    # 2-1. keywords/phrases percentage of all words
    f.write('Feature 2-1: percentage of all keywords from all words' + '\n')
    f.write(str(t.keywordPer))
    f.write('\n')
    f.write('========================================================================================' + '\n')

    # 3. question marks, i.e. interrogative sentences 
    # 3-1. appearing times
    # 3-2. appearing locations
    # 3-3. percentage of all sentences
    f.write('Feature 3-1: questionmarks count' + '\n')
    f.write(str(t.questionStat))
    f.write('\n')
    f.write('========================================================================================' + '\n')
    f.write('Feature 3-2: questionmarks location' + '\n')
    f.write(str(t.questionLoc))
    f.write('\n')
    f.write('========================================================================================' + '\n')
    f.write('Feature 3-3: percentage of sentences ending with questionmark from all sentences' + '\n')
    f.write(str(t.questionPer))
    f.write('\n')
    f.write('========================================================================================' + '\n')
    

    # 4. speech length
    # 4-1. with metric of words with stopwords
    # 4-2. with metric of words without stopwords
    # 4-3. with metric of sentences
    f.write('Feature 4-1: speech length in metric of words with stopwords' + '\n')
    f.write(str(t.transSize))
    f.write('\n')
    f.write('========================================================================================' + '\n')
    f.write('Feature 4-2: speech length in metric of words without stopwords' + '\n')
    f.write(str(t.transSizeNS))
    f.write('\n')
    f.write('========================================================================================' + '\n')
    f.write('Feature 4-3: speech length in metric of sentences' + '\n')
    f.write(str(t.sentNum))
    f.write('\n')
    f.write('========================================================================================' + '\n')

    # 5. keyword with questionmarks
    # 5-1. keyword num with Q, keyword num all, keyword percentage with Q
    f.write('Feature 5-1: keywords with questionmarks (num with questionmarks, num all, percentage with questionmarks)' + '\n')
    f.write(str(t.keywordQOutput))
    f.write('\n')
    f.write('========================================================================================' + '\n')

    # 6. non speech terms
    f.write('Feature 6-1: non speech terms' + '\n')
    f.write(str(t.nSpeech))
    f.write('\n')
    f.write('========================================================================================' + '\n')

    # 7. turn length
    f.write('Feature 7-1: length by turn' + '\n')
    f.write(str(t.listTurn))

    f.close()


def lengthByTurnInMat():
    import numpy as np
    import scipy.io as sio
    fileName = ["Transcript1.txt","Transcript3.txt","Transcript4.txt","Transcript5.txt","Transcript6.txt","Transcript7.txt"]
    dataLengthTurn = defaultdict(dict)
    for n in range(len(fileName)):
        t = KeywordCount(fileName[n])
        t.updateWordCount()
        t.updateKeywordStatistics()
        t.questionMark()
        t.emoVoice()
        t.keywordQuestionmark()
        t.nonSpeech()
        t.lengthByTurn()
        ID, Len = [], []
        for i in t.listTurn:
            ID.append(0 if i[0]=='p' else 1)
            Len.append(i[1])
        turnArray = np.array([ID, Len])
        dataLengthTurn['T'+str(n+1)] = turnArray
        #print(turnArray)
    sio.savemat('lengthByTurn.mat', dataLengthTurn)
    # print(turnArray)

def outputPersonal(fname):
    t = LabelledKeywordCount(fname,'p')
    t.updateWordCount()
    t.updateKeywordStatistics()
    t.questionMark()
    t.emoVoice()
    t.keywordQuestionmark()

    f = open('FeaturesOutput_'+t.name+'_Label_'+t.label,'w')
    #file information
    f.write('File name: ' + t.name + '\n')
    f.write('========================================================================================' + '\n')
    f.write('Object: ' + t.label + '\n')
    f.write('========================================================================================' + '\n')
    
    f.write('========================================================================================' + '\n')
    f.write('========================================================================================' + '\n')
    f.write('========================================================================================' + '\n')
    
    # 1. keywords & key phrases of each domain
    # 1-1. appearing times(sorted)
    f.write('Feature 1-1: times(int) certain keyword appears in the transcript ' + '\n')
    f.write(str(t.keywordStatSorted))
    f.write('\n')
    f.write('========================================================================================' + '\n')
    #1-2. appearing locations(sequenced)
    f.write('Feature 1-2: locations(list of floats) certain keyword appears in the transcrip' + '\n')
    f.write(str(t.keywordLoc))
    f.write('\n')
    f.write('========================================================================================' + '\n')
    # f.write('Feature 3: percentage per keyword from all keywords' + '\n')
    # f.write(str(t.keywordPart))
    # f.write('\n')
    # f.write('========================================================================================' + '\n')

    # 2. keywords & key phrases overall
    # 2-1. keywords/phrases percentage of all words
    f.write('Feature 2-1: percentage of all keywords from all words' + '\n')
    f.write(str(t.keywordPer))
    f.write('\n')
    f.write('========================================================================================' + '\n')

    # 3. question marks, i.e. interrogative sentences 
    # 3-1. appearing times
    # 3-2. appearing locations
    # 3-3. percentage of all sentences
    f.write('Feature 3-1: questionmarks count' + '\n')
    f.write(str(t.questionStat))
    f.write('\n')
    f.write('========================================================================================' + '\n')
    f.write('Feature 3-2: questionmarks location' + '\n')
    f.write(str(t.questionLoc))
    f.write('\n')
    f.write('========================================================================================' + '\n')
    f.write('Feature 3-3: percentage of sentences ending with questionmark from all sentences' + '\n')
    f.write(str(t.questionPer))
    f.write('\n')
    f.write('========================================================================================' + '\n')
    

    # 4. speech length
    # 4-1. with metric of words with stopwords
    # 4-2. with metric of words without stopwords
    # 4-3. with metric of sentences
    f.write('Feature 4-1: speech length in metric of words with stopwords' + '\n')
    f.write(str(t.transSize))
    f.write('\n')
    f.write('========================================================================================' + '\n')
    f.write('Feature 4-2: speech length in metric of words without stopwords' + '\n')
    f.write(str(t.transSizeNS))
    f.write('\n')
    f.write('========================================================================================' + '\n')
    f.write('Feature 4-3: speech length in metric of sentences' + '\n')
    f.write(str(t.sentNum))
    f.write('\n')
    f.write('========================================================================================' + '\n')

    # 5. keyword with questionmarks
    # 5-1. keyword num with Q, keyword num all, keyword percentage with Q
    f.write('Feature 5-1: keywords with questionmarks (num with questionmarks, num all, percentage with questionmarks)' + '\n')
    f.write(str(t.keywordQOutput))

    f.close()
