
# Author: peter.zhong@au1.ibm.com
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the Apache 2.0 License.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Apache 2.0 License for more details.
import time
import re
import ast
import json
import distance
from apted import APTED, Config
from itertools import product
from apted.helpers import Tree
from lxml import etree, html
from collections import deque
#from parallel import parallel_process
from tqdm import tqdm
from zss import simple_distance, Node
import string
from typing import Any, Callable, Optional, Sequence
import numpy as np
import Levenshtein
import editdistance


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.


class TEDS(object):
    ''' Tree Edit Distance basead Similarity
    '''
    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (n_jobs >= 1), 'n_jobs must be an integer greather than 1'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        ''' Converts HTML tree to the format required by apted
        '''
        global __tokens__
        if node.tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(node.tag,
                                 int(node.attrib.get('colspan', '1')),
                                 int(node.attrib.get('rowspan', '1')),
                                 cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        ''' Computes TEDS score between the prediction and the ground truth of a
            given sample
        '''
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        
        if pred.xpath('body/table') and true.xpath('body/table'):
            pred = pred.xpath('body/table')[0]
            true = true.xpath('body/table')[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0

def convert_table_to_html_str(table_row_list=[]):
    """
    Given a list of table rows, build the corresponding html string, which is used to compute the TEDS score.
    We use the official code of PubTabNet to compute TEDS score, it does not consider '<th>' label.
    We also remove unneccessary spaces within a table cell and extra '\n' as they will influence the TEDS score.
    """
    html_table_str = "<html><body><table>" + '\n'
    for data_row in table_row_list:
        html_table_str += "<tr>"
        for cell_str in data_row:
            html_table_str += f"<td>{cell_str}</td>"
        html_table_str += "</tr>"
        html_table_str += '\n'
    html_table_str += "</table></body></html>"
    html_table_str = html_table_str.replace('\n','')
    return html_table_str
def convert_markdown_table_to_html(markdown_table):
    """
    Converts a markdown table to the corresponding html string for TEDS computation.
    """
    # remove extra code block tokens like '```markdown' and '```
    markdown_table = markdown_table.strip('```markdown').strip('```').strip() 
    row_str_list = markdown_table.split('\n')
    # extra the first header row and other data rows
    valid_row_str_list = [row_str_list[0]]+row_str_list[2:]
    table_rows = []
    for row_str in valid_row_str_list:
        one_row = []
        for cell in row_str.strip().split('|')[1:-1]:
            if set(cell) != set(' '):
                one_row.append(cell.strip())
            else:
                one_row.append(' ')
        table_rows.append(one_row)
    # build html string based on table rows
    html_str = convert_table_to_html_str(table_rows)
    return html_str
def dict_to_html(data):
    html = "<html><body><table>\n"
    for key, value in data.items():
        if not isinstance(value, str):
            value = str(value)
        value_str = ' '.join(value)
        
        html += f"  <tr><td>{key}</td><td>{value_str}</td></tr>\n"
    html += "</table></body></html>"
    return html

def convert_str_to_dict(predict_str: str):
    """
    Parses the 'predict' string and returns a dictionary.
    Missing or unparseable content is handled gracefully.

    Parameters:
    - predict_str (str): The prediction string containing the output dict.

    Returns:
    - dict: A dictionary extracted from the predict string.
    """
    # Remove code fences like ```python\n...\n```
    code_fence_pattern = r'```(?:python|json)?\n(.*?)\n```'
    match = re.search(code_fence_pattern, predict_str, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1)
    else:
        content = predict_str.strip()

    data = {}
    success = False

    # try parsing with JSON
    try:
        data = json.loads(content)
        success = True
    except json.JSONDecodeError:
        pass

    # try parsing with ast.literal_eval
    if not success:
        try:
            data = ast.literal_eval(content)
            if isinstance(data, dict):
                success = True
        except (ValueError, SyntaxError):
            pass

    # try parsing with regex
    if not success:
        key_value_pattern = r'["\']?([\w\s]+)["\']?\s*[:=]\s*["\']?([^\n,"\'{}]+)["\']?'
        matches = re.findall(key_value_pattern, content)
        try:
            for key, value in matches:
                data[key.strip()] = value.strip()
        except:
            return {}

    if not data:
        return {}

    try:
        result = {k.strip(): str(v).strip() for k, v in data.items()}
    except:
        return {}
    return result


def convert_str_to_multi_dict(predict_str: str):
    """
    Parses the 'predict' string and returns a dictionary.
    Handles nested dictionaries and missing or unparseable content gracefully.

    Parameters:
    - predict_str (str): The prediction string containing the output dict.

    Returns:
    - dict: A dictionary extracted from the predict string.
    """
    # Remove code fences like ```python\n...\n```
    code_fence_pattern = r'```(?:python|json)?\n(.*?)\n```'
    matches = re.findall(code_fence_pattern, predict_str, re.DOTALL | re.IGNORECASE)
    if matches:
        content = max(matches, key=len)
    else:
        content = predict_str.strip()
    
    def strip_variable_assignment(s):
        variable_assignment_pattern = r'^\s*\w+\s*=\s*'
        return re.sub(variable_assignment_pattern, '', s.strip(), count=1)

    content = strip_variable_assignment(content)

    def remove_comments(s):
        return re.sub(r'#.*', '', s)

    content = remove_comments(content)

    last_brace_pos = content.rfind('}')
    if last_brace_pos != -1:
        content = content[:last_brace_pos+1]

    data = {}
    success = False

    # try parsing with ast.literal_eval
    try:
        data = ast.literal_eval(content)
        if isinstance(data, dict):
            success = True
    except (ValueError, SyntaxError, TypeError):
        pass

    if not success:
        return {}

    def process_data(obj):
        if isinstance(obj, dict):
            return {k: process_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [process_data(elem) for elem in obj]
        else:
            return obj

    data = process_data(data)

    return data


def generate_combinations(input_dict):
    """
    Function to generate all possible combinations of values from a dictionary.
    """
    kie_answer = input_dict
    if not isinstance(kie_answer, dict):
        kie_answer = kie_answer.strip('"')
        try:
            kie_answer = json.loads(kie_answer)
        except json.JSONDecodeError:
            try:
                kie_answer = ast.literal_eval(kie_answer)
                if not isinstance(kie_answer, dict):
                    kie_answer = ast.literal_eval(kie_answer)
            except (ValueError, SyntaxError):
                print(f"Unable to parse 'answers' field: {kie_answer}")
                return {}
        
        # Ensure the parsed result is a dictionary.
        if not isinstance(kie_answer, dict):
            print("Parsed 'answers' is still not a dictionary.")
            raise ValueError("Input could not be parsed into a dictionary.")
    
        keys = list(kie_answer.keys())
        
        value_lists = []
        for single_key in keys:
            sinlge_value = kie_answer[single_key]
            if not isinstance(sinlge_value, list):
                sinlge_value = [sinlge_value]
            value_lists.append(sinlge_value)
    
        # Compute the Cartesian product of the value lists.
        combinations = list(product(*value_lists))
    
        # Create a dictionary for each combination of values.
        result = [dict(zip(keys, values)) for values in combinations]

        return result
    
    else:
        keys = list(input_dict.keys())
        value_lists = [input_dict[key] for key in keys]

        # Compute the Cartesian product of the value lists.
        combinations = list(product(*value_lists))

        # Create a dictionary for each combination of values.
        result = [dict(zip(keys, values)) for values in combinations]

        return result


def compute_f1_score(preds, gts, ignores=[]):
    """Compute the F1-score for KIE task between predicted and ground truth dictionaries.

    Args:
        preds (dict): The predicted key-value pairs.
        gts (dict): The ground truth key-value pairs.
        ignores (list): The list of keys to ignore during evaluation.

    Returns:
        dict: A dictionary where keys are field names and values are their corresponding F1-scores.
    """
    # Optionally remove ignored keys from predictions and ground truths
    keys = set(preds.keys()).union(set(gts.keys())) - set(ignores)
    f1_scores = {}

    for key in keys:
        pred_value = preds.get(key, None)
        gt_value = gts.get(key, None)

        if pred_value:
            pred_value = pred_value.lower().strip().replace("\n"," ").replace(" ", "")
        if gt_value:
            gt_value = gt_value.lower().strip().replace("\n"," ").replace(" ", "")

        if pred_value is None and gt_value is None:
            continue
        elif pred_value is None:
            precision = 0.0
            recall = 0.0
        elif gt_value is None:
            # false positive
            precision = 0.0
            recall = 0.0
        else:
            if pred_value == gt_value:
                # True positive
                precision = 1.0
                recall = 1.0
            else:
                precision = 0.0
                recall = 0.0

        # Compute F1-score
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores[key] = f1_score

    if len(f1_scores) == 0:
        return 0
    average_f1 = sum(f1_scores.values()) / len(f1_scores)

    return average_f1


def pre_clean(text):
    text = re.sub(r'<bos>|<eos>|<pad>|<unk>', '', text)
    text = re.sub(r'\s##(\S)', r'\1', text)
    text = re.sub(r'\\\s', r'\\', text)
    text = re.sub(r'\s\*\s\*\s', r'**', text)
    text = re.sub(r'{\s', r'{', text)
    text = re.sub(r'\s}', r'}', text)
    text = re.sub(r'\s}', r'}', text)
    text = re.sub(r'\\begin\s', r'\\begin', text)
    text = re.sub(r'\\end\s', r'\\end', text)
    text = re.sub(r'\\end{table}', r'\\end{table} \n\n', text)
    text = text.replace('\n', ' ')
    text = text.replace('*', ' ')
    text = text.replace('_', ' ')
    return text


def get_tree(input_str):
    tree = (Node('ROOT').addkid(Node('TITLE')))

    lines = input_str.split("\n")
    lines = [pre_clean(line) for line in lines]
    last_title = ''
    for line in lines:
        if line.startswith('#'):
            child = tree.get('ROOT')
            line = line.replace('#', '')
            child.addkid(Node(line))
            last_title = line
        else:
            if last_title == '':
                child = tree.get('TITLE')
                child.addkid(Node(line))
            else:
                child = tree.get(last_title)
                child.addkid(Node(line))
    return tree

def STEDS(pred_tree, ref_tree):
    def my_distance(pred, ref):
        if len(pred.split()) == 0 or len(ref.split()) == 0:
            return 1
        else:
            return 0
    total_distance = simple_distance(pred_tree, ref_tree, label_dist=my_distance)
    num_of_nodes = max(len(list(pred_tree.iter())), len(list(ref_tree.iter())))
    return 1-total_distance/num_of_nodes


def doc_parsing_evaluation(pred, gt):
    score = 0
    if not isinstance(pred, str):
        return 0
    pred_tree = get_tree(pred)
    gt_tree = get_tree(gt)
    score = STEDS(pred_tree, gt_tree)

    return score


def wrap_html_table(html_table):
    """
    The TEDS computation from PubTabNet code requires that the input html table should have <html>, <body>, and <table> tags.
    Add them if they are missing.
    """
    html_table = html_table.replace('\n','')
    # add missing <table> tag if missing
    if "<table" in html_table and "</table>" not in html_table:
        html_table = html_table + "</table>"
    elif "<table" not in html_table and "</table>" in html_table:
        html_table = "<table>" + html_table
    elif "<table" not in html_table and "</table>" not in html_table:
        html_table = "<table>" + html_table + "</table>"
    else:
        pass
    # add <body> and <html> tags if missing
    if '<body>' not in html_table:
        html_table = '<body>' + html_table + '</body>'
    if '<html>' not in html_table:
        html_table = '<html>' + html_table + '</html>'
    return html_table
    

def get_anls(s1, s2):
    try:
        s1 = s1.lower()
        s2 = s2.lower()
    except:
        pass
    if s1 == s2:
        return 1.0
    iou = 1 - editdistance.eval(s1, s2) / max(len(s1), len(s2))
    anls = iou
    return anls


def ocr_eval(references,predictions):
    socre_=0.0
    None_num=0
    for idx,ref_value in enumerate(references):
        pred_value = predictions[idx]
        pred_values, ref_values = [], []
        if isinstance(pred_value, str):
            pred_values.append(pred_value)
        else:
            pred_values = pred_value
        if isinstance(ref_value, str):
            ref_values.append(ref_value)
        else:
            ref_values = ref_value
        
        temp_score = 0.0
        temp_num = len(ref_values)
        
        for tmpidx, tmpref in enumerate(ref_values):
            tmppred = pred_values[tmpidx] if tmpidx < len(pred_values) else pred_values[0]
            if len(pred_values) == 1 and tmppred != "None" and "None" not in ref_values:  # pred 1, and not None
                temp_score = max(temp_score, get_anls(tmppred, tmpref))
                temp_num = len(ref_values)
            else:
                if tmppred=='None' and tmpref!='None':
                    temp_score += 0.0
                elif tmpref=='None':
                    temp_num -= 1
                else:
                    temp_score += get_anls(tmppred, tmpref)
        if temp_num == 0:
            ocr_score = 0.0
            None_num += 1
        else:
            ocr_score = temp_score / (temp_num)
        socre_ += ocr_score
    if None_num == len(references):
        return 9999
    else:
        return round(socre_ / (len(references)-None_num), 5)

def process_predictions(input_path, output_path):
    with open(input_path, "r") as f:
        predict_file = json.load(f)

    teds = TEDS(n_jobs=32)

    task_type_list = ["table parsing en", "chart parsing en", "document parsing en"]

    res_data_list = []
    time1=time.time()
    for index, data_item in enumerate(tqdm(predict_file)):
        
        
        if data_item["type"] == "table parsing en":
            if type(data_item["answers"])==list and len(data_item["answers"]) == 1:
                if not isinstance(data_item["predict"], str):
                    data_item["score"] = 0
                elif not isinstance(data_item["question"], str):
                    data_item["ignore"] = "True"
                    data_item["score"] = 0

                elif "html" in data_item["question"].lower():
                    no_find = False
                    predict_table = data_item["predict"].replace('\n','')
                    if "<body" in predict_table:
                        predict_table = re.findall('<body.*', predict_table)[0]
                    elif "<table" in predict_table:
                        predict_table = re.findall('<table.*', predict_table)[0]
                    else:
                        no_find = True

                    if no_find:
                        data_item["score"] = 0
                    else:
                        pred_table_html = wrap_html_table(predict_table)
                        gold_table_html = wrap_html_table(data_item["answers"][0])
                        try:
                            data_item["score"] = teds.evaluate(pred_table_html, gold_table_html)
                        except:
                            data_item["score"] = 0

                elif "markdown" in data_item["question"].lower():
                    if not isinstance(data_item["predict"], str):
                        
                        prediction = str(data_item["predict"])
                        pred_table_html = convert_markdown_table_to_html(prediction)
                        gt_table_html = convert_markdown_table_to_html(data_item["answers"][0])
                        data_item["score"] = teds.evaluate(pred_table_html, gt_table_html)

                    else:
                        pred_table_html = convert_markdown_table_to_html(data_item["predict"])
                        gt_table_html = convert_markdown_table_to_html(data_item["answers"][0])
                        data_item["score"] = teds.evaluate(pred_table_html, gt_table_html)
            else:
                raise ValueError


        elif data_item["type"] == "chart parsing en":
            answer = data_item["answers"][0]
            if data_item["predict"]:

                pred_chart_dict = convert_str_to_multi_dict(data_item["predict"])
                if len(pred_chart_dict) == 0:
                    data_item["score"] = 0
                else:
                    pred_chart_html = dict_to_html(pred_chart_dict)
                    gt_chart_html = dict_to_html(answer)
                    data_item["score"] = teds.evaluate(pred_chart_html, gt_chart_html)
            else:
                data_item["score"] = 0

        elif data_item["type"] == "document parsing en":
            assert type(data_item["answers"])==list and len(data_item["answers"]) == 1
            data_item["score"] = doc_parsing_evaluation(data_item["predict"], data_item["answers"][0])
        res_data_list.append(data_item)

    for task_name in task_type_list:
        print("\n" + task_name)
        mean_score, total_len = 0, .0
        for item in res_data_list:
            if item["type"] == task_name:
                total_len += 1
                mean_score += item["score"]
        
        mean_score = mean_score / total_len if total_len > 0 else 0
        print(f"Task {task_name}, total instructions: {total_len}, average score: {mean_score:.3f}\n")
    time2=time.time()
    print(time2-time1)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(predict_file, file, ensure_ascii=False, indent=4)
process_predictions("/home/lrz/reward/test.json","/home/lrz/reward/output.json")