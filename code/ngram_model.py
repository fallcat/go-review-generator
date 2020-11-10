import math
import random
import copy
import re
import string
from string import punctuation
from collections import Counter
from collections import defaultdict
from queue import Queue
from collections import deque
import statistics

################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    
    res = []

    if n < 0:
        return None

    if n == 0:
        curr_text = copy.deepcopy(text)
        for char in curr_text:
            res.append(("",char))
    else:
        temp = copy.deepcopy(text)
        for index,item in enumerate(temp):
            context = []
            counter = n
            if index == 0:
                res.append((start_pad(n),item))
                # while counter > 0:
                #     context.append(start_pad(1))
                #     counter -= 1
                # res.append(("".join(context),item))
            else:
                while counter > 0:
                    if index - counter < 0:
                        context.append(start_pad(1))
                        counter -= 1
                    else:
                        context.append(temp[index - counter])
                        counter -= 1
                res.append(("".join(context),item))

    return res



def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.order_n = n
        self.ksmooth = k
        self.ngrams = defaultdict(lambda: Counter())
        self.vocab = Counter()

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return set(self.vocab.keys())

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        curr_ngrams = ngrams(self.order_n, text)
        self.vocab += Counter(text)
        self.vocab.pop(" ", None)


        for context,char in curr_ngrams:
            tok_dict = self.ngrams[context]
            tok_dict += Counter(char)
            self.ngrams[context] = tok_dict

    def add_file_lines(self, path):
        '''update the model with all of the lines in a whole file'''
        ## copied code from method: def create_ngram_model(model_class, path, n=2, k=0) above
        with open(path, encoding='utf-8', errors='ignore') as f:
            self.update(f.read())


    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''

        V = len(self.get_vocab())

        if len(self.ngrams[context]) == 0:
            return 1/V

        total = sum(self.ngrams[context].values())

        return (self.ngrams[context].get(char,0) + self.ksmooth) / (total + self.ksmooth*V)

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''

        chars = sorted(self.get_vocab())
        cmltv_sum = 0
        rnum = random.random()

        print("rnum: ", rnum)

        for c in chars:
            print("Line 136: ", self.prob(context,c))
            cmltv_sum += self.prob(context,c)
            if cmltv_sum >= rnum:
                return c

    def random_text(self, length):

        context = Queue()
        context_window = self.order_n
        result = [] 
        # import pdb;pdb.set_trace()      

        for i in range(self.order_n):
            context.put(start_pad(1))

        for i in range(length):
            print("Context for each character: ", list(context.queue))
            curr_tok = self.random_char("".join(list(context.queue)))
            print("Line 152: ", curr_tok)
            if curr_tok == None:
                context.queue.clear()
                for i in range(self.order_n):
                    context.put(start_pad(1))
                continue
            else:
                context.put(curr_tok)
                if len(list(context.queue)) > self.order_n:
                    context.get() 

            result.append(curr_tok)

        return "".join(result)

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''

        curr_ngrams = ngrams(self.order_n, text)

        log_sum = 0
        m_root = 1/(len(text))

        for context,tok in curr_ngrams:
            if self.prob(context,tok) == 0:
                return float('inf')
            log_sum += math.log(1/self.prob(context,tok))     

        return math.exp(m_root*log_sum)

## ADDED
def write_output_file(filename, results):

    ## results as a string
    ## write to output file
    with open(filename,'w') as f:
        f.write(results + '\n')

    f.close()

## ADDED
def write_file(filename, results):

    ## results as string, very specific format
    ## write to output file
    with open(filename,'w') as f:
        for i in range(0,len(results),2):
            f.write("string: " + str(results[i]) + "   perplexity:" + str(results[i+1]) + "\n")

    f.close()

## ADDED
def write_out_results(filename, results):

    
    ## write to output file
    ## results as a list
    with open(filename,'w') as f:
        for item in results:
            f.write(str(item) + '\n')

    f.close()

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        self.order_n = n
        self.ksmooth = k
        self.ngrams = defaultdict(lambda: Counter())
        self.vocab = Counter()
        self.models = self.create_models()
        self.lambdas = [1/(n+1)]*(n+1)

    def get_vocab(self):
        res = set()
        for m in self.models:
            res.update(m.get_vocab())

        return res

    def update(self, text):
        for m in self.models:
            m.update(text)

    def update_file_to_vocab(self, filename):
        for m in self.models:
            m.add_file_lines(filename)

    def prob(self, context, char):
        """lambas: percentage value weight to that model's probabilities"""

        
        ## Iterate through models to get probabilities of each model*lambdaweight
        interp = 0

        for i, model in enumerate(self.models):

            ## for order zero
            if i == 0:
                temp_context = ""
            ## for orders greater than zero
            else:
                temp_context = context[:i]

            interp += self.lambdas[i]*model.prob(temp_context,char)

        return interp

    def create_models(self):
        temp = []
        for order in range(self.order_n + 1):
            temp.append(NgramModel(order,self.ksmooth))

        return temp

    def create_lambdas(self):
        temp = []
        for order in range(self.order_n + 1):
            temp.append(1/(self.order_n + 1))

        self.lambdas = temp

    def update_lambdas(self, new_lambda_values):
        if len(new_lambda_values) == self.order_n:
            self.lambdas = new_lambda_values
        else:
            return -1

    def inter_perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''

        inter_perplex = []

        for i, model in self.models:

            curr_ngrams = ngrams(self.order_n, text)

            log_sum = 0
            m_root = 1/(len(text))

            for context,tok in curr_ngrams:
                if self.prob(context,tok) == 0:
                    return float('inf')
                log_sum += math.log(1/self.prob(context,tok))  


            inter_perlex.append(math.exp(m_root*log_sum))

        return statistics.mean(inter_perlex)

def get_accuracy(data1,data2):

    if len(data1) != len(data2):
        return -1

    correct = 0
    for i in range(len(data1)):
        if data1[i] == data2[i]:
            correct += 1

    return correct / len(data1)



################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################