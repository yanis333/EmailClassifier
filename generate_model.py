import os
import re

CLASSES = ['ham', 'spam']
SMOOTHING_DELTA = 0.5

def get_parsed_training_set(training_dir):
    parsed_training_set = []
    for root, _, training_set in os.walk(training_dir, topdown=True):
        for file in training_set:
            with open(os.path.join(root, file)) as f:
                email_contents = f.read().lower()
                new_words = re.split('[^a-zA-Z]+', email_contents)
                classification = re.findall('ham|spam', file)[0]
                for word in new_words:
                    parsed_training_set.append((word, classification))

    return parsed_training_set

def get_vocabulary(training_set):
    return set([word for word, _ in training_set])

def get_class_frequencies(training_set):
    class_frequencies = {}
    for entry in parsed_training_set:
        _, classification = entry
        if class_frequencies.get(classification) is None:
            class_frequencies[classification] = 1
        else:
            class_frequencies[classification] += 1

    return class_frequencies

def get_word_frequencies(training_set):
    frequencies = {}
    for classification in CLASSES:
        frequencies[classification] = {}
    
    for entry in training_set:
        word, classification = entry
        if frequencies[classification].get(word) is None:
            frequencies[classification][word] = 1
        else:
            frequencies[classification][word] += 1

    return frequencies



def get_word_conditional(word, classification, smoothing=0):
    word_frequency = word_frequencies.get(classification).get(word)
    if word_frequency is None:
        word_frequency = 0

    # print('frequency of word {} in class {}: {}'.format(word, classification, word_frequency))  
    smoothed_word_frequency = word_frequency + smoothing
    class_frequency = class_frequencies.get(classification)

    conditional = smoothed_word_frequency / (class_frequency + (smoothing * vocabulary_size))

    return smoothed_word_frequency, conditional


if __name__ == '__main__':
    parsed_training_set = get_parsed_training_set('./training_set')
    vocabulary = get_vocabulary(parsed_training_set)
    word_frequencies = get_word_frequencies(parsed_training_set)
    class_frequencies = get_class_frequencies(parsed_training_set)

    model = []
    for word in vocabulary:
        vocabulary_size = len(vocabulary)

        model_entry = [word]
        for classification in CLASSES:
            smoothed_word_frequency, conditional = get_word_conditional(word, classification, SMOOTHING_DELTA)
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
