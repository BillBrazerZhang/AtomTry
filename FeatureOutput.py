from FeatureExtract import KeywordCount 
    def outputFeatures(fname):

        t1 = KeywordCount(fname)
        t1.updateWordCount()
        t1.updateKeywordStatistics()
        t1.emoVoice()
        f = open('Feature'+t1.name,'w')
        #file information
        f.write('File name: ' + t1.name + '\n')
        f.write('========================================================================================' + '\n')
        f.write('Object: ' + 'Overall' + '\n')
        f.write('========================================================================================' + '\n')
        f.write('Keyword Dictionary:' + '\n')
        f.write(str(t1.keywordCTS))
        f.write('\n')
        f.write('========================================================================================' + '\n')
        f.write('========================================================================================' + '\n')
        f.write('========================================================================================' + '\n')
        # 1. keywords & key phrases of each domain
        # 1-1. appearing times(sorted)
        f.write('Feature 1-1: times(int) certain keyword appears in the transcript ' + '\n')
        f.write(str(t1.keywordStatSorted))
        f.write('\n')
        f.write('========================================================================================' + '\n')
        #1-2. appearing locations(sequenced)
        f.write('Feature 1-2: locations(list of floats) certain keyword appears in the transcrip' + '\n')
        f.write(str(t1.keywordLoc))
        f.write('\n')
        f.write('========================================================================================' + '\n')
        # f.write('Feature 3: percentage per keyword from all keywords' + '\n')
        # f.write(str(t1.keywordPart))
        # f.write('\n')
        # f.write('========================================================================================' + '\n')

        # 2. keywords & key phrases overall
        # 2-1. keywords/phrases percentage of all words
        f.write('Feature 2-1: percentage of all keywords from all words' + '\n')
        f.write(str(t1.keywordPer))
        f.write('\n')
        f.write('========================================================================================' + '\n')

        # 3. question marks, i.e. interrogative sentences 
        # 3-1. appearing times
        # 3-2. appearing locations
        # 3-3. percentage of all sentences
        f.write('Feature 3-1: questionmarks count' + '\n')
        f.write(str(t1.questionStat))
        f.write('\n')
        f.write('========================================================================================' + '\n')
        f.write('Feature 3-2: questionmarks location' + '\n')
        f.write(str(t1.questionLoc))
        f.write('\n')
        f.write('========================================================================================' + '\n')
        f.write('Feature 3-3: percentage of sentences ending with questionmark from all sentences' + '\n')
        f.write(str(t1.questionPer))
        f.write('\n')
        f.write('========================================================================================' + '\n')
        

        # 4. speech length
        # 4-1. with metric of words with stopwords
        # 4-2. with metric of words without stopwords
        # 4-3. with metric of sentences
        f.write('Feature 4-1: speech length in metric of words with stopwords' + '\n')
        f.write(str(t1.transSize))
        f.write('\n')
        f.write('========================================================================================' + '\n')
        f.write('Feature 4-2: speech length in metric of words without stopwords' + '\n')
        f.write(str(t1.transSizeNS))
        f.write('\n')
        f.write('========================================================================================' + '\n')
        f.write('Feature 4-3: speech length in metric of sentences' + '\n')
        f.write(str(t1.sentNum))
        f.write('\n')
        f.close()