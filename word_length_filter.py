from filter import Filter

class WordLengthFilter(Filter):
    def apply(self, parsed_training_set):
        print('applying', self.__class__)
        filtered = []
        for entry in parsed_training_set:
            word, _ = entry
            if len(word)>2 and len(word) < 9 :
                filtered.append(entry)

        return filtered