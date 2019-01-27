WARNINGS = False
LOW_WARNINGS = False
VERBOSE = False


class WordCategory:
    def __init__(self, token_order, word, lemma, token, pos, gender, number, status, person, tense):
        self.token_order = token_order
        self.word = word
        self.lemma = lemma
        self.token = token
        self.pos = pos
        self.gender = gender
        self.number = number
        self.status = status
        self.person = person
        self.tense = tense

    @staticmethod
    def _is_number(str):
        try:
            float(str)
            return True
        except:
            return False

    @staticmethod
    def hebrew_fixer(str):
        res = ""
        for c in str.strip():
            if c.isalpha():
                res += c
        return res

    @staticmethod
    def number_fixer(str):
        res = ""
        for c in str.strip():
            if c.isnumeric():
                res += c
        return res

    def equal_bio(self, bio, categories):
        if len(categories.categories) > 0:
            if self.token_order == categories.categories[0].token_order:
                return True
            else:
                return False
        if WordCategory.hebrew_fixer(self.token) == WordCategory.hebrew_fixer(bio.word.strip()):
            return True
        if WordCategory.hebrew_fixer(self.token) == WordCategory.hebrew_fixer(bio.word.replace("-", "")):
            return True
        if self.token == "###NUMBER###" or self.token == "###NUMEXP###":
            if WordCategory._is_number(WordCategory.number_fixer(bio.word)):
                return True
        return False

    def __str__(self):
        return f"{self.token_order} {self.word} {self.lemma} {self.token} {self.pos} {self.gender} {self.number} {self.status} {self.person} {self.tense}"

    @staticmethod
    def parse_category(str):
        splitted = str.strip().split(" ")
        if len(splitted) > 10 and LOW_WARNINGS:
            print(f"Warning: {str} is larger than 10 columns")
        elif len(splitted) < 10:
            raise RuntimeError(f"Invalid Category: {str}")
        return WordCategory(splitted[0], splitted[1], splitted[2], splitted[3], splitted[4], splitted[5],
                            splitted[6], splitted[7], splitted[8], splitted[9])


class WordCategories:
    def __init__(self):
        self.categories = []

    def add_category(self, category):
        self.categories.append(category)


class WordBIO:
    def __init__(self, word, bio):
        self.word = word.strip()
        self.bio = bio.strip()

    def __str__(self):
        return f"{self.word} {self.bio}"

    @staticmethod
    def parse_bio(str):
        splitted = str.split(" ")
        if len(splitted) != 2:
            raise RuntimeError(f"Invalid bio line: {str}")
        return WordBIO(splitted[0], splitted[1])


class WordCategoryBio:
    def __init__(self, is_space, is_word, token_order, word, bio, lemma, token, pos, gender, number, status, person, tense, prefixes_and_suffixes=False, prefix_one="", prefix_two="", prefix_three="", suffix_one="", suffix_two="", suffix_three=""):
        self.is_space = is_space
        self.is_word = is_word
        self.bio = bio
        self.token_order = token_order
        self.word = word
        self.lemma = lemma
        self.token = token
        self.pos = pos
        self.gender = gender
        self.number = number
        self.status = status
        self.person = person
        self.tense = tense
        self.prefixes_and_suffixes = prefixes_and_suffixes
        self.prefix_one = prefix_one
        self.prefix_two = prefix_two
        self.prefix_three = prefix_three
        self.suffix_one = suffix_one
        self.suffix_two = suffix_two
        self.suffix_three = suffix_three

    @staticmethod
    def create_space():
        return WordCategoryBio(True, False, None, None, None, None, None, None, None, None, None, None, None)

    @staticmethod
    def create_word(token_order, word, bio, lemma, token, pos, gender, number, status, person, tense):
        return WordCategoryBio(False, True, token_order, word, bio, lemma, token, pos, gender, number, status, person, tense)

    @staticmethod
    def create_word_extended(token_order, word, bio, lemma, token, pos, gender, number, status, person, tense, prefix_one, prefix_two, prefix_three, suffix_one, suffix_two, suffix_three):
        return WordCategoryBio(False, True, token_order, word, bio, lemma, token, pos, gender, number, status, person, tense, True, prefix_one, prefix_two, prefix_three, suffix_one, suffix_two, suffix_three)

    def _get_prefix(self):
        if self.prefix_one == "no_pref":
            return "no_pref"
        pref = self.prefix_one
        if self.prefix_two != "no_pref":
            pref += self.prefix_two
        if self.prefix_three != "no_pref":
            pref += self.prefix_three
        return pref

    def _get_suffix(self):
        if self.suffix_one == "no_suffix":
            return "no_suffix"
        suff = self.suffix_one
        if self.suffix_two != "no_suffix":
            suff += self.suffix_two
        if self.suffix_three != "no_suffix":
            suff += self.suffix_three
        return suff

    def __str__(self):
        if self.is_space:
            return ""
        elif not self.prefixes_and_suffixes:
            return f"{self.word} {self.bio} {self.token_order} {self.lemma} {self.token} {self.pos} {self.gender} {self.number} {self.status} {self.person} {self.tense}"
        else:
            pref = self._get_prefix()
            suff = self._get_suffix()
            return f"{self.word} {self.bio} {self.token_order} {self.lemma} {self.token} {self.pos} {self.gender} {self.number} {self.status} {self.person} {self.tense} {pref} {suff}"

class CategoryBioFileMerger:
    def __init__(self, category_file, bio_file):
        self.category_file = category_file
        self.bio_file = bio_file

    @staticmethod
    def _get_word_categories(cf, bio, category):
        categories = WordCategories()
        last_category = None
        misses_count = 0

        if VERBOSE:
            print("*******************************")
            print(f"Merging {bio.word}")

        while True:
            if category is None:
                cf_line = cf.readline()
                if cf_line is None and len(categories) == 0:
                    raise RuntimeError(f"Failed to parse category for {bio.word} - End of stream")
                elif cf_line is None:
                    break
                if cf_line.strip() == "":
                    if len(categories.categories) == 0:
                        raise RuntimeError(f"Failed to parse category for {bio.word} - End of stream")
                    break
                category = WordCategory.parse_category(cf_line)
            if not category.equal_bio(bio, categories):
                misses_count += 1
                if len(categories.categories) == 0:
                    if WARNINGS:
                        print(f"Warning: Ignoring {category}")
                    if misses_count > 20:
                        raise RuntimeError(f"Probably lost {bio}")
                    category = None
                    continue
                else:
                    last_category = category
                    break
            categories.add_category(category)
            last_category = category
            category = None

        return categories, last_category

    @staticmethod
    def _get_main_category(categories):
        len_to_cat = {}
        longest_cat = None
        max_length = None
        for category in categories:
            if longest_cat is None:
                longest_cat = category
                max_length = len(category.word)
            len_to_cat[len(category.word)] = category
        for key in len_to_cat.keys():
            if key >= max_length:
                max_length = key
                longest_cat = len_to_cat[key]
        return longest_cat


    def _merge(self, cf, bf, ignore, merge_partial_tokens):
        next_category = None


        while True:
            bf_line = bf.readline()
            if bf_line is None or bf_line == "":
                return
            if bf_line.strip() == "":
                yield WordCategoryBio.create_space()
                continue
            bio = WordBIO.parse_bio(bf_line)
            if bio.word in ignore:
                print(f"Explicit ignore: Ignoring {bio.word}")
                continue
            categories, next_category = CategoryBioFileMerger._get_word_categories(cf, bio, next_category)
            prefixes = ["no_pref", "no_pref", "no_pref"]
            suffixes = ["no_suffix", "no_suffix", "no_suffix"]
            main_category = CategoryBioFileMerger._get_main_category(categories.categories)
            saving_prefixes = True
            pre_suf_counter = 0
            for category in categories.categories:
                if not merge_partial_tokens:
                    yield WordCategoryBio.create_word(category.token_order, category.word, bio.bio, category.lemma,
                                                      category.token, category.pos, category.gender, category.number,
                                                      category.status, category.person, category.tense)
                else:
                    if main_category.word == category.word:
                        pre_suf_counter = 0
                        saving_prefixes = False
                    elif saving_prefixes:
                        prefixes[pre_suf_counter] = category.word
                        pre_suf_counter += 1
                    else:
                        suffixes[pre_suf_counter] = category.word
                        pre_suf_counter += 1
                    if merge_partial_tokens and pre_suf_counter > 3:
                        raise RuntimeError("merge_partial_tokens and pre_suf_counter >= 3")

            if merge_partial_tokens:
                if main_category is None:
                    raise RuntimeError("main_category is None")
                yield WordCategoryBio.create_word_extended(main_category.token_order, main_category.word, bio.bio,
                                                           main_category.lemma, main_category.token, main_category.pos,
                                                           main_category.gender, main_category.number, main_category.status,
                                                           main_category.person, main_category.tense,
                                                           prefixes[0], prefixes[1], prefixes[2],
                                                           suffixes[0], suffixes[1], suffixes[2])

    def merge(self, ignore=[], merge_partial_tokens=False):
        with open(self.category_file) as cf:
            with open(self.bio_file) as bf:
                for category_bio in self._merge(cf, bf, ignore, merge_partial_tokens):
                    yield category_bio



with open("o", "w") as f:
    merger = CategoryBioFileMerger("c", "t")
    for category_bio in merger.merge(["Décor", "Economics/National", "CRNT:NASDAQ", "עלCNN", "הCIA"], True):
        f.writelines([str(category_bio) + "\n"])

