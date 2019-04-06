import os
import sys
import re
import math
from filter import Filter
from stop_word_filter import StopWordFilter
from word_length_filter import WordLengthFilter

class EmailClassifier:

    def __init__(self, classes, filters=[]):
        self._classes = classes
        self._filters = filters

    def generate_model(self, training_set_dir):
        if not hasattr(self, '_model'):
            self._model = EmailClassifier._ModelGenerator(training_set_dir, self._classes, self._filters)
            self._model_value = self._model.generate_model()
    
    def generate_test_result(self,test_dir):
        self._test = EmailClassifier._TestGenerator(test_dir, self._classes, self._filters)
        self._test.generate_test(self._model_value)
        return


    def with_filter(self, f: Filter):
        self._filters.append(f)
        return self


    class _TestGenerator:

        def __init__(self, test_dir, classes, filters):
            self._test_dir = test_dir
            self._classes = classes
            self._filters = filters

        def generate_test(self, model_value):
            parsed_test_result = []
            if '' in model_value:
                del model_value['']

            for root, _, training_set in os.walk(self._test_dir, topdown=True):
                for file in training_set:
                    with open(os.path.join(root, file), encoding='ISO-8859-1') as f:
                        email_contents = f.read()
                        file_words = re.split('[^a-zA-Z]', email_contents)
                        file_words_lower = [x.lower() for x in file_words]
                        file_words_set = set(file_words_lower)
                        spam_total = 0
                        ham_total = 0
                        classification = re.findall('|'.join(self._classes), file)[0]

                        for word in file_words_set:
                            if word in model_value and word is not '':
                                spam_total = spam_total + math.log10(model_value[word][4])
                                ham_total = ham_total + math.log10(model_value[word][2])

                        test_classification = 'ham'
                        if spam_total > ham_total :
                            test_classification = 'spam'

                        right_or_wrong = 'wrong'
                        if test_classification == classification :
                            right_or_wrong ='right'
                        
                        parsed_test_result.append((file, test_classification, ham_total, spam_total, classification, right_or_wrong))
            

            type_model = './baseline-result.txt'
            for f in self._filters:
                if f.name == 'StopWordPath':
                    type_model = './stopword-result.txt'
                if f.name == 'WordLengthFilter':
                    type_model = './wordlength-result.txt'


            with open(type_model, 'w') as f:
                for i in range(len(parsed_test_result)):
                    entry = parsed_test_result[i]
                    f.write('{}  '.format(i+1))
                    for x in entry:
                        f.write('{}  '.format(x))

                    f.write('\n')     
      
            return


    class _ModelGenerator:

        def __init__(self, training_set_dir, classes, filters):
            self._model = None
            self._classes = classes
            self._filters = filters

            self._parsed_training_set = self.get_parsed_training_set(training_set_dir)
            self._vocabulary = self.get_vocabulary()
            self._word_frequencies = self.get_word_frequencies()
            self._class_frequencies = self.get_class_frequencies()
            self._smoothing_delta = 0.5

        def get_parsed_training_set(self, training_dir):
            # parse training set into the following format: (word, classification)

            parsed_training_set = []
            for root, _, training_set in os.walk(training_dir, topdown=True):
                for file in training_set:
                    with open(os.path.join(root, file)) as f:
                        email_contents = f.read().lower()
                        new_words = re.split('[^a-zA-Z]', email_contents)
                        classification = re.findall('|'.join(self._classes), file)[0]
                        for word in new_words:
                            parsed_training_set.append((word, classification))

            # apply any filters here
            if len(self._filters) > 0:
                print('length of parsed training set before filtering:', len(parsed_training_set))
                for f in self._filters:
                    parsed_training_set = f.apply(parsed_training_set)

                print('length of parsed training after before filtering:', len(parsed_training_set), '\n')

            return parsed_training_set

        def get_vocabulary(self):
            '''
            returns set of words in the vocabulary
            '''
            return set([word for word, _ in self._parsed_training_set])

        def get_class_frequencies(self):
            '''
            returns a dict containing the frequency of each class

            e.g. {
                class1: freq1,
                class2: freq2,
                ...
            }
            '''
            class_frequencies = {}
            for entry in self._parsed_training_set:
                _, classification = entry
                if class_frequencies.get(classification) is None:
                    class_frequencies[classification] = 1
                else:
                    class_frequencies[classification] += 1

            return class_frequencies

        def get_word_frequencies(self):
            '''
            returns a dict containing the frequency of each word in the vocabulary, by class

            e.g. {
                class1: {
                    word1: freq1,
                    word2: freq2,
                    ...
                },
                class2: {
                    word1: freq1,
                    word2: freq2,
                    ... 
                },
                ...
            }
            '''
            frequencies = {}
            for classification in self._classes:
                frequencies[classification] = {}
            
            for entry in self._parsed_training_set:
                word, classification = entry
                if frequencies[classification].get(word) is None:
                    frequencies[classification][word] = 1
                else:
                    frequencies[classification][word] += 1

            return frequencies



        def get_word_conditional(self, word, classification, smoothing=0):
            '''
            returns a tuple containing the smoothed word frequency and the conditional probability given the class
            '''
            word_frequency = self._word_frequencies.get(classification).get(word)
            if word_frequency is None:
                word_frequency = 0

            # print('frequency of word {} in class {}: {}'.format(word, classification, word_frequency))  
            smoothed_word_frequency = word_frequency + smoothing
            class_frequency = self._class_frequencies.get(classification)
            vocabulary_size = len(self._vocabulary)
            conditional = smoothed_word_frequency / (class_frequency + (smoothing * vocabulary_size))

            return smoothed_word_frequency, conditional

        def generate_model(self):
            '''
            generates the model 
            '''
            if self._model is not None:
                return

            model = []
            for word in self._vocabulary:
                model_entry = [word]
                for classification in self._classes:
                    smoothed_word_frequency, conditional = self.get_word_conditional(word, classification, self._smoothing_delta)
                    # print('conditional probability P({}|{}): {}'.format(word, classification, conditional))
                    model_entry.append(smoothed_word_frequency)
                    model_entry.append(conditional)

                model.append(tuple(model_entry))

            model = sorted(model, key=lambda entry: entry[0])


            model_value = {}
            for element in model:
                model_value[element[0]] = element

            type_model = './model.txt'
            for f in self._filters:
                if f.name == 'StopWordPath':
                    type_model = './stopword-model.txt'
                if f.name == 'WordLengthFilter':
                    type_model = './wordlength-model.txt'

            with open(type_model, 'w') as f:
                for i in range(len(model)):
                    entry = model[i]
                    f.write('{}  '.format(i+1))
                    for x in entry:
                        f.write('{}  '.format(x))

                    f.write('\n')
            return model_value

if __name__ == '__main__':
    TRAINING_SET_DIR = sys.argv[1]
    TEST_SET_DIR = sys.argv[2]
    
    classes = ['ham', 'spam']
    classifier1 = EmailClassifier(classes)
    classifier1.generate_model(TRAINING_SET_DIR)
    classifier1.generate_test_result(TEST_SET_DIR)

    classifier2 = EmailClassifier(classes, [WordLengthFilter()])
    classifier2.generate_model(TRAINING_SET_DIR)
    classifier2.generate_test_result(TEST_SET_DIR)

    classifier3 = EmailClassifier(classes, [StopWordFilter('./english_stop_words.txt')])
    classifier3.generate_model(TRAINING_SET_DIR)
    classifier3.generate_test_result(TEST_SET_DIR)
