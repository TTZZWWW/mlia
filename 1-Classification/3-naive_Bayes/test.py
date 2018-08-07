
import re

big_string = open('email/spam/%d.txt' % 25).read()
list_of_tokens = re.split(r'\W*', big_string)
word_list = [tok.lower() for tok in list_of_tokens if len(tok) > 2]
print(word_list)

big_string = open('email/ham/%d.txt' % 25).read()
list_of_tokens = re.split(r'\W*', big_string)
word_list = [tok.lower() for tok in list_of_tokens if len(tok) > 2]
print(word_list)