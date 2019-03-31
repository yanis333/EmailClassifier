import os
import re
from filter import Filter
from stop_word_filter import StopWordFilter

class EmailClassifier:

    def __init__(self, classes):
        self._classes = classes
        self._filters = []

    def generate_model(self, training_set_dir):
        if not hasattr(self, '_model'):
            self._model = EmailClassifier._ModelGenerator(training_set_dir, self._classes, self._filters)
            self._model.generate_model()

    def with_filter(self, f: Filter):
        self._filters.append(f)
        return self

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

                print('length of parsed training after before filtering:', len(parsed_training_set))

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
            
            with open('./model.txt', 'w') as f:
                for i in range(len(model)):
                    entry = model[i]
                    f.write('{}  '.format(i+1))
                    for x in entry:
                        f.write('{}  '.format(x))

                    f.write('\n')

if __name__ == '__main__':
    classes = ['ham', 'spam']
    classifier = EmailClassifier(classes)
    classifier.generate_model('./training_set')
