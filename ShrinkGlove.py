import sys

common_words = set()
word_list = open(sys.argv[3], encoding='utf-8')
for word in word_list:
    common_words.add(word.strip('\n'))
word_list.close()

glove = {}
big_glove = open(sys.argv[1], encoding='utf-8')
little_glove = open(sys.argv[2], 'w', encoding='utf-8')
for line in big_glove:
    values = line.split()
    word = values[0]
    if word in common_words:
        little_glove.write(line)
big_glove.close()
little_glove.close()
