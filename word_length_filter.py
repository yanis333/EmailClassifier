from filter import Filter

class WordLengthFilter(Filter):

    def __init__(self):
        super().__init__()
        self.name = "WordLengthFilter"

    def apply(self, parsed_training_set):
        print('applying', self.__class__)
        filtered = []
        for entry in parsed_training_set:
            word, _ = entry
            if len(word)>2 and len(word) < 9 :
                filtered.append(entry)

        return filtered