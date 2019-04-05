from filter import Filter

class StopWordFilter(Filter):

    

    def __init__(self, stop_words_path):
        super().__init__()
        self.name = "StopWordPath"
        self._stop_words = self._get_stop_words(stop_words_path)

    def apply(self, parsed_training_set):
        print('applying', self.__class__)
        filtered = []
        for entry in parsed_training_set:
            word, _ = entry
            if word not in self._stop_words:
                filtered.append(entry)

        return filtered

    def _get_stop_words(self, stop_words_path):
        stop_words = []
        with open(stop_words_path, 'r') as f:
            for line in f.readlines():
                stop_words.append(line.strip())

        return stop_words