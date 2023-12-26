"""
No Doc
"""
import re
import sys
from typing import Dict
import warnings
# from collections import Counter
from json import loads, JSONDecodeError

import numpy as np
from lexrank import LexRank
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from tqdm import tqdm


warnings.simplefilter("ignore")


class IterJsonl:
    """
    None doc string
    """
    def __init__(self, path: str) -> None:
        self.file = open(path,
                         "r",
                         encoding="utf-8-sig")
        self.loads = loads

    def __iter__(self):
        print('start')
        return self

    def __next__(self) -> Dict:

        try:
            line = self.file.readline()
            if line == '':
                raise StopIteration
            return line

        except JSONDecodeError as err:
            self.file.close()
            print("stop iter")
            raise StopIteration("get end of file") from err

    def __del__(self):
        self.file.close()
        del self.file

    @staticmethod
    def clear_text(string: str, fltr=r"[^а-яё А-ЯЁ.,]") -> str:
        """
            Принимает на вход строку, удаляет исмвол конца строки
            возвращает строку
        """
        return re.sub(fltr, "", string)


class DataStorage():
    """_summary_
    """
    def __init__(self, path: str,
                 col_name: str,
                 auto_add=False,
                 task: str = None,
                 save_path: str = None,):

        self.post_data = dict()
        self.coment_data = dict()
        self.col_name = col_name
        self.path = path
        self._id_stor = None
        self._item = None
        self.loads = loads
        self.col_parent = "parent_id"
        self.task = task
        self.save_path = save_path
        if auto_add:
            self.__auto_add_data_()

    def __auto_add_data_(self):
        count = 0
        for line_item in IterJsonl(self.path):
            try:
                data = self.loads(line_item)
                self._add_(data)
                count += 1
            except JSONDecodeError:
                pass

        print(f"Download {count} lines")

    def __insert_post_data_(self, key_store: str, line_item: Dict) -> None:
        self.post_data[key_store] = line_item

    def __insert_coment_data_(self, key_store: str, line_item: Dict) -> None:
        """_summary_

        Args:
            key_store (str): root_id
            line_item (Dict): dict from json
        """

        for item in line_item:
            try:
                (self.coment_data[key_store]
                 [item].append(line_item[item]))
            except KeyError:
                self.coment_data[key_store] =\
                    dict(
                        [(i, [line_item[i],]) for i in line_item.keys()])
                break

    def _add_(self, line_item) -> None:
        try:
            key_store = line_item[self.col_name]
            self.__insert_coment_data_(key_store, line_item)
        except KeyError:
            key_store = line_item["id"]
            self.__insert_post_data_(key_store, line_item)


class IterDataStorage(DataStorage):
    """
    no docstring
    """

    def __init__(self, path: str,
                 col_name: str,
                 col_target: str = "text",
                 auto_add: bool = True,
                 task: str = None,
                 save_path: str = None,
                 ) -> None:
        super().__init__(path, col_name,
                         auto_add, task,
                         save_path)
        self.col_target = col_target
        self.start_val = 0
        self.stop_val = len(self.post_data)
        self.keys = list(self.post_data.keys())
        self.sentences = list()
        self.hash_comment = []
        self.hash_post = ''
        self.summary = []
        self.result = []
        self.open = open

    def __iter__(self):
        return self

    def save_file(self):
        """
        No Doc
        """
        import jsonlines
        with jsonlines.open(self.save_path, mode='w') as writer:
            for i in self.result:
                writer.write(i)

    def final(self):
        """
        no doc
        """
        summarizer_lsa = LsaSummarizer()
        text = '. '.join(self.summary)
        parser = PlaintextParser.from_string(
            text,
            Tokenizer("russian"))
        summary = summarizer_lsa(parser.document, 5)
        self.summary = ' '.join([str(sentence) for sentence in summary])
        self.result.append(
            {"summary": self.summary,
             "post_hash": self.hash_post,
             "comments_hash": self.hash_comment})

    def get_embed(self, text: list, post_type: bool = True) -> np.array:
        """_summary_
        Args:
            text (list): list of data text
        Returns:
            np.array: emmbedings
        """
        post, sentences = text
        try:
            lxr = LexRank([post], keep_numbers=True,
                          keep_emails=False, keep_urls=False)
        except ValueError:
            self.hash_comment = []
            self.summary = []
            return None
        # tf_scores = [
        # Counter(lxr.tokenize_sentence(sentence)) for sentence in sentences]
        lex_scores = lxr.rank_sentences(sentences, threshold=.01)
        ret = []

        hash_comment = []
        if post_type:
            mi = np.quantile(lex_scores, .5)
            ma = np.quantile(lex_scores, .75)
            mask = np.logical_and(ma < lex_scores, lex_scores > mi)
            iters = np.argsort(lex_scores)[mask]
        else:
            mi = np.quantile(lex_scores, .1)
            ma = np.quantile(lex_scores, .2)
            mask = np.logical_and(mi < lex_scores, lex_scores > ma)
            iters = np.argsort(lex_scores)[mask]
        for i in iters:

            ret.append(sentences[i])
            hash_comment.append(self.coment_data[
                self.keys[self.start_val]]["hash"][i])

        sentences = ret

        tmp = []
        for i in ret:
            tmp.append(lxr.sentences_similarity(post, i))
        tmp = np.argsort(np.array(tmp))
        self.summary = []
        self.sentences = []
        for i in tmp[-(len(tmp)//2) + 2:]:
            self.sentences.append(ret[i])
            self.hash_comment.append(
                hash_comment[i])
        return 1

    def lexrank(self, documents,
                threshold=.01):
        """
        no doc
        """
        try:
            lxr = LexRank(documents=documents, keep_numbers=True,
                          keep_emails=False, keep_urls=False)
        except ValueError:
            self.hash_comment = []
            self.summary = []
            return None

        lex_scores = lxr.rank_sentences(
                self.sentences,
                threshold=threshold)
        lex_scores_arg = np.argsort(lex_scores)
        lex_scores = np.sort(lex_scores)
        perc = 0.1
        score = np.quantile(lex_scores, perc)
        res = []
        count = 0
        while count < len(lex_scores):
            if lex_scores[count] >= score:
                res.extend(lex_scores_arg[count-1: count + 2])
                count += 1
                perc += 0.15
                if perc >= 1:
                    res.extend(lex_scores_arg[-4:])
                    break
                score = np.quantile(lex_scores, perc)
            count += 1

        self.summary = []
        if self.hash_comment:
            tmp_hash, self.hash_comment = self.hash_comment, []
        else:
            tmp_hash = self.coment_data[
                self.keys[self.start_val]]["hash"]
        for ind in set(res):
            self.summary.append(self.sentences[ind])
            self.hash_comment.append(tmp_hash[ind])

    def get_comment_all(self, threshold=.01):
        """
        no doc
        """
        keys_ = self.keys[self.start_val]
        self.sentences = self.coment_data[keys_][self.col_target]
        self.hash_post = self.post_data[keys_]["hash"]
        self.hash_comment = self.coment_data[keys_]["hash"]
        #  for doc in self.coment_data[keys_][self.col_target]:
        #      self.sentences.extend(PunktSentenceTokenizer().tokenize(doc))
        self.lexrank(documents=self.sentences,
                     threshold=threshold)
        self.final()
        return f'post_hash: {self.hash_post}'

    def other_comments(self, threshold=.01, post_type: bool = True):
        """
        no doc
        """

        keys_ = self.keys[self.start_val]
        post = self.post_data[keys_][self.col_target]
        self.hash_post = self.post_data[keys_]["hash"]
        self.sentences = self.coment_data[keys_][self.col_target]
        #  for doc in self.coment_data[keys_][keys_][self.col_target]:
        #      self.sentences.extend(PunktSentenceTokenizer().tokenize(doc))
        self.get_embed([post, self.sentences], post_type)

        if len(self.sentences) <= 30:
            # print(len(self.sentences))
            # print(len(self.hash_comment))
            self.summary = [i for i in self.sentences]
            self.final()
            return f'"summary": {self.summary}, "post_hash": {self.hash_post}'
        self.lexrank(documents=self.sentences,
                     threshold=threshold)
        self.final()
        return f'"summary": {self.summary}, "post_hash": {self.hash_post}'

    def get_job(self):
        """
        no doc
        """
        if self.task == "all_comments":
            self.get_comment_all(threshold=.01)
        elif self.task == "topic_comments":
            self.other_comments(threshold=.01, post_type=False)
        elif self.task == "post_comments":
            self.other_comments(threshold=.01, post_type=True)

    def __next__(self) -> Dict:

        if self.start_val < self.stop_val:
            self.get_job()
            self.start_val += 1
        else:
            raise StopIteration
        return "StopIteration"


if __name__ == '__main__':

    iterator = IterDataStorage(
        col_name="root_id",
        path=(sys.argv[2]
              if len(sys.argv) >= 2
              else "./dataset.jsonl"),
        task=(sys.argv[1]
              if len(sys.argv) >= 3
              else 'topic_comments'),
        save_path=(sys.argv[3]
                   if len(sys.argv) >= 4
                   else './out_topic.jsonl'))
    for i in tqdm(iterator.post_data.keys()):
        next(iterator)
    iterator.save_file()
