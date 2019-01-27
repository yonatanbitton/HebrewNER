import os
import pandas as pd
import xml.etree.ElementTree as ET
import json

from tei_reader import TeiReader

BIO_TAGS = {'I_MISC__AFF', 'I_ORG', 'B_MISC__AFF', 'B_PERS', 'I_PERS', 'B_DATE', 'I_PERCENT', 'I_MISC__ENT',
            'I_MONEY', 'B_MISC__ENT', 'B_TIME', 'I_LOC', 'I_TIME', 'B_ORG', 'B_LOC', 'I_MISC_EVENT', 'I_DATE'}

BASIC_BIO_TAGS = {'MISC', 'PERCENT', 'LOC', 'TIME', 'ORG', 'PERS', 'DATE', 'MONEY'}


def corpus_to_text():
    data_path = "resources" + os.sep + "tagged_corpus_naama.txt"
    with open(data_path, 'r', encoding='utf8') as f:
        all_text = []
        lines = f.readlines()
        for line in lines:
            all_text.append(line.split(" ")[0])

    all_text_str = " ".join(all_text)
    data_str_path = "resources" + os.sep + "corpus_text_naama.txt"
    with open(data_str_path, 'w', encoding='utf8') as f:
        f.write(all_text_str)

def get_all_tags():
    data_path = "resources" + os.sep + "tagged_corpus_naama.txt"
    with open(data_path, 'r', encoding='utf8') as f:
        all_tags = set()
        all_basic_tags = set()
        lines = f.readlines()
        for line in lines:
            if len(line.split(" ")) > 1:
                bio_tag = line.split(" ")[1].rstrip("\n")
                if len(line.split("_")) > 1:
                    basic_tag = line.split("_")[1]
                    all_basic_tags.add(basic_tag.rstrip("\n"))
                all_tags.add(bio_tag)

    print(all_tags)
    print(all_basic_tags)

def split_tag_to_basic(tag):
    if len(tag.split("_")) > 1:
        return tag.split("_")[1].rstrip("\n")
    else:
        return tag

def bio_to_biluo():
    data_path = "resources" + os.sep + "tagged_corpus_naama.txt"
    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for i in range(1, len(lines) - 1):
            prev_tag, curr_tag, next_tag = get_tags(i, lines)
            new_tag = curr_tag
            if split_tag_to_basic(curr_tag) in BASIC_BIO_TAGS:
                if split_tag_to_basic(prev_tag) != split_tag_to_basic(curr_tag) and split_tag_to_basic(curr_tag) != split_tag_to_basic(next_tag):
                    new_tag = 'U-' + split_tag_to_basic(curr_tag)
                else:
                    if split_tag_to_basic(prev_tag) == split_tag_to_basic(curr_tag):
                        if split_tag_to_basic(curr_tag) != split_tag_to_basic(next_tag):
                            new_tag = 'L-' + split_tag_to_basic(curr_tag)
                        else: #split_tag_to_basic(curr_tag) == split_tag_to_basic(next_tag)
                            new_tag = 'I-' + split_tag_to_basic(curr_tag)
                    else: # split_tag_to_basic(prev_tag) != split_tag_to_basic(curr_tag) BUT split_tag_to_basic(curr_tag) != split_tag_to_basic(next_tag)
                        new_tag = 'B-' + split_tag_to_basic(curr_tag)
            lines[i] = lines[i].rstrip("\n") + " " + new_tag + "\n"

    new_tag_path = "resources" + os.sep + "biluo.txt"
    with open(new_tag_path, 'w', encoding='utf8') as f:
        f.write(' '.join(lines))


    print("DONE")


def get_tags(i, lines):
    prev_line = lines[i - 1]
    curr_line = lines[i]
    next_line = lines[i + 1]

    prev_tag = get_tag(prev_line)
    curr_tag = get_tag(curr_line)
    next_tag = get_tag(next_line)

    return prev_tag, curr_tag, next_tag


def get_tag(line):
    line_tag = 'O'
    if len(line.split(" ")) > 1:
        line_tag = line.split(" ")[1].rstrip("\n")
    return line_tag

def get_word(line):
    x = line.split(" ")[0]
    if 'DOCSTART' in x:
        return 'DOCSTART'
    elif '-' in x:
        return x.strip("-")
    elif '\u200f' in x:
        return x.split('\u200f')[1]
    return x



def add_bio_tags_to_processed_file():
    processed_data_path = "resources" + os.sep + "corpus_text_naama_processed_trimmed.txt"
    processed_data = read_file(processed_data_path)

    tags_data_path = "resources" + os.sep + "tagged_corpus_naama_trimmed.txt"
    tags_data = read_file(tags_data_path)

    tags_index = 0

    for processed_line_idx, processed_line in enumerate(processed_data):
        processed_line_parts = processed_line.split(" ")
        new_parts = processed_line_parts
        curr_tag_line = tags_data[tags_index]
        if "שלנו" in curr_tag_line:
            curr_tag_line = curr_tag_line.replace(r'"', '')
        if "החברה" in get_word(curr_tag_line) or 'האג"ח' in get_word(curr_tag_line):
            curr_tag_line = curr_tag_line.replace(':', '')
        if processed_line_idx == 77949 or processed_line_idx == 80000:
            print("HE")
        if (get_word(curr_tag_line) in new_parts and get_word(curr_tag_line) != '') or new_parts[1] == '43341' or new_parts[1] == '52819':
            by_word = get_word(curr_tag_line)
            given_tag = get_tag(curr_tag_line)
        elif '-' in tags_data[tags_index + 1] and '-' in new_parts and not 'DOCSTART' in old_processed_line_parts:
            tags_index += 1
            curr_tag_line = tags_data[tags_index]
            given_tag = curr_tag_line
            by_word = '-'
            tags_index += 1
        elif by_word == 'השנים' and new_parts[2] == '###NUMBER###':
            tags_index += 1
            curr_tag_line = tags_data[tags_index]
            given_tag = curr_tag_line
            by_word = get_word(curr_tag_line)
        elif new_parts[2] == '-' or new_parts[2] == '%' or new_parts[1] == '43338' or new_parts[1] == '43339':
            given_tag = old_tag
        elif new_parts[0] == '36972' or new_parts[0] == '36973':
            given_tag = curr_tag_line
            by_word = get_word(curr_tag_line)
        else:
            found_relevant_tag = False
            while not found_relevant_tag:
                tags_index += 1
                curr_tag_line = tags_data[tags_index]
                if len(curr_tag_line.split(" "))>1:
                    found_relevant_tag = True
            if new_parts[2] == '###NUMBER###' or (get_word(curr_tag_line) in new_parts or (new_parts[5] == 'preposition' and new_parts[2] in get_word(curr_tag_line)) or get_word(curr_tag_line)[:-1] in new_parts):
                by_word = get_word(curr_tag_line)
                given_tag = get_tag(curr_tag_line)

        new_parts.append(given_tag)
        new_parts.append(by_word)
        updated_line = " ".join([x.rstrip("\n") for x in new_parts]) + "\n"
        print("Updated line:")
        print(updated_line)
        print(processed_line_idx)
        processed_data[processed_line_idx] = updated_line
        old_tag = given_tag
        old_processed_line_parts = processed_line_parts

    print("Yo")


def read_file(processed_data_path):
    with open(processed_data_path, 'r', encoding='utf8') as f:
        processed_data = f.readlines()
    return processed_data


def merged_output_to_pandas_csv():
    with open('resources' + os.sep + 'merged_output', 'r', encoding='utf8') as f:
        merged_input = f.readlines()
        rows_list = []
    for line in merged_input:
        parts = line.split(" ")
        if len(parts) != 13:
            continue
        line_d = {'Word': parts[0], 'Bio': parts[1], 'TokenOrder': parts[2], 'Lemma': parts[3], 'Token': parts[4],
                  'Pos': parts[5], 'Gender': parts[6], 'Number': parts[7], 'Status': parts[8], 'Person': parts[9],
                  'Tense': parts[10], 'Prefix': parts[11].rstrip("\n"), 'Suffix': parts[12].rstrip("\n")}
        rows_list.append(line_d)
    df = pd.DataFrame(rows_list)
    print(f"merged df at shape: {df.shape}")
    df.to_csv("resources" + os.sep + "dataset.csv", index=False)


def get_tags_df(i, lines):
    prev_line = lines.iloc[i - 1]
    curr_line = lines.iloc[i]
    next_line = lines.iloc[i + 1]

    prev_tag = prev_line['Bio']
    curr_tag = curr_line['Bio']
    next_tag = next_line['Bio']

    return prev_tag, curr_tag, next_tag

def merged_bio_to_biluo():
    dataset = pd.read_csv("resources" + os.sep + "dataset.csv")
    for i in range(0, len(dataset)-1):
        prev_tag, curr_tag, next_tag = get_tags_df(i, dataset)
        new_tag = curr_tag
        if split_tag_to_basic(curr_tag) in BASIC_BIO_TAGS:
            if split_tag_to_basic(prev_tag) != split_tag_to_basic(curr_tag) and split_tag_to_basic(
                    curr_tag) != split_tag_to_basic(next_tag):
                new_tag = 'U-' + split_tag_to_basic(curr_tag)
            else:
                if split_tag_to_basic(prev_tag) == split_tag_to_basic(curr_tag):
                    if split_tag_to_basic(curr_tag) != split_tag_to_basic(next_tag):
                        new_tag = 'L-' + split_tag_to_basic(curr_tag)
                    else:  # split_tag_to_basic(curr_tag) == split_tag_to_basic(next_tag)
                        new_tag = 'I-' + split_tag_to_basic(curr_tag)
                else:  # split_tag_to_basic(prev_tag) != split_tag_to_basic(curr_tag) BUT split_tag_to_basic(curr_tag) != split_tag_to_basic(next_tag)
                    new_tag = 'B-' + split_tag_to_basic(curr_tag)
        dataset.loc[dataset.index[i], 'BILUO'] = new_tag

    dataset.to_csv("resources" + os.sep + "dataset_biluo.csv", index=False)
    print("DONE")

def xml_to_csv3():
    path = 'resources' + os.sep + 'yael_corpus' + os.sep + 'jsons' + os.sep + 'ab_yehushua.json'
    with open(path, encoding='utf-8') as fh:
        data = json.load(fh)
    print("d")

def xml_to_csv():
    path = 'resources' + os.sep + 'yael_corpus' + os.sep + 'ab_yehushua.xml'
    tree = ET.parse(path)
    all_text = ""
    for elem in tree.iter():
        tag = elem.tag.split("}")[1]
        print(tag, " - ", elem.text)
        if elem.text:
            all_text += elem.text


def xml_to_csv2():
    path = 'resources' + os.sep + 'yael_corpus' + os.sep + 'ab_yehushua.xml'

    reader = TeiReader()
    corpora = reader.read_file(path)  # or read_string
    print(corpora.text)


    # tree = ET.parse(path)
    # print("Yo")
    # root = tree.getroot()
    # ans_dict = {}
    # for node in root.iter():
    #     tag = node.tag.split("}")[1]
    #     print(tag)
    #     print(node.text)
    #     print("\n")
    #     # if tag == 'title':
    #     #     ans_dict['title'] = node.text
    #     # if tag == 'singer':
    #     #     ans_dict['singer'] = node.text
    #     # if tag == 'writer':
    #     #     ans_dict['writer'] = node.text
    #     # if tag == 'composer':
    #     #     ans_dict['composer'] = node.text
    #     # if tag == 'album':
    #     #     ans_dict['album'] = node.text
    #     # if tag == 'publisher':
    #     #     ans_dict['publisher'] = node.text
    #
    # return ans_dict


    # from tei_reader import TeiReader
    # reader = TeiReader()
    # corpora = reader.read_file('resources' + os.sep + 'yael_corpus' + os.sep + 'ab_yehushua.xml')  # or read_string
    # print(corpora.text)
    #
    # show element attributes before the actual element text
    # print(corpora.tostring(lambda x, text: str(list(a.key + '=' + a.text for a in x.attributes)) + text))


if __name__ == '__main__':
    # corpus_to_text()
    # get_all_tags()
    # bio_to_biluo()
    # add_bio_tags_to_processed_file()
    # merged_output_to_pandas_csv()
    merged_bio_to_biluo()
    # xml_to_csv()



