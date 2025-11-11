import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
import re
import random

def integers_to_onehot(integers, max_value=10):
    integers = [min(i, max_value) for i in integers]
    onehot_matrix = torch.zeros((len(integers), max_value + 1))
    for i, integer in enumerate(integers):
        onehot_matrix[i, integer] = 1
    return onehot_matrix

def get_labels(path):
    data = pd.read_excel(path, index_col=0)
    data = data.dropna(axis=0, how='any')
    labels, convert_dict = {}, {}

    for i in range(5):
        convert_dict[i] = i * 0.15 + 0.2
    for name in data[1:].index.values:
        labels[name] = data.loc[name].values.tolist()[:-1]
        temp = []
        for i in labels[name]:
            temp.append(int(i))
        labels[name] = temp
        labels[name] = [convert_dict[min(int(i), 4)] for i in labels[name]]
    return labels

def extract_string(string):
    pattern = r'《(.*?)》'
    matches = re.findall(pattern, string)
    return matches

def get_paint_to_course(paint_to_course, path):
    df = pd.read_excel(path, index_col=0, skiprows=1)
    df = df.fillna("empty")
    for index, row in df.iterrows():
        if row[0] == 'empty':
            continue
        paint_name, course_name = index, row[0]
        course_name = extract_string(course_name)[0]
        paint_to_course[paint_name] = course_name
    return paint_to_course

def get_dimension_to_idx(dimension_to_idx, path):
    data = pd.read_excel(path, index_col=0)
    data = data.dropna(axis=0, how='any')
    column_names = data.iloc[0]
    for idx in range(len(column_names)):
        column_names[idx] = column_names[idx].replace(' ', '')
        dimension_to_idx[column_names[idx]] = idx
    return dimension_to_idx

def get_all_reflect_relation(data_path):
    # 修改路径拼接为 os.path.join，避免路径问题
    path = os.path.join(data_path, '打分表')
    paint_to_course, dimension_to_idx = {}, {}
    
    for filename in os.listdir(path):
        deep_path = os.path.join(path, filename)
        if '表' in filename:
            paint_to_course = get_paint_to_course(paint_to_course, deep_path)

    dimension_to_idx = get_dimension_to_idx(dimension_to_idx, os.path.join(data_path, '画', '打分1.xlsx'))

    DF = pd.read_excel(os.path.join(data_path, "1.xlsx"), header=None)
    DF = DF.fillna("empty")
    first_row = DF.iloc[0, :-1].tolist()
    second_row = DF.iloc[1, :-1].tolist()

    ability = {}
    pos = 0
    while pos < len(first_row):
        while pos < len(first_row) and first_row[pos] == str('empty'):
            pos += 1
        idx = pos + 1
        ability[first_row[pos]] = []
        ability[first_row[pos]].append(second_row[pos].replace(" ", ""))
        while idx < len(first_row) and first_row[idx] == str('empty'):
            ability[first_row[pos]].append(second_row[idx].replace(" ", ""))
            idx += 1
        pos = idx

    dimension_adj_matrix = torch.zeros(77, 77).cuda()
    for ke, va in ability.items():
        for i in range(len(va)):
            for j in range(i + 1, len(va), 1):
                i_idx, j_idx = dimension_to_idx[va[i]], dimension_to_idx[va[j]]
                dimension_adj_matrix[i_idx, j_idx], dimension_adj_matrix[j_idx, i_idx] = 1, 1

    return paint_to_course, dimension_adj_matrix

def get_RL_data(image_size, data_path):
    # 修改路径拼接为 os.path.join，避免路径问题
    image_path = os.path.join(data_path, '画')
    image_list, labels = {}, {}
    paint_to_course, _ = get_all_reflect_relation(data_path)

    for filename in os.listdir(image_path):
        deep_path = os.path.join(image_path, filename)
        if filename.endswith('.xlsx'):
            labels.update(get_labels(deep_path))
        if not os.path.isdir(deep_path):
            continue
        for wholefilename in os.listdir(deep_path):
            if wholefilename.endswith('.jpg') or wholefilename.endswith('.png') or wholefilename.endswith('.jpeg'):
                img_path = os.path.join(deep_path, wholefilename)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((image_size, image_size))
                img = torch.tensor(np.array(img)).permute(2, 0, 1).float()
                name_without_extension = os.path.splitext(wholefilename)[0]
                if name_without_extension in paint_to_course.keys():
                    image_list[name_without_extension] = img

    image_list_temp = {}
    for ke in image_list.keys():
        if ke not in labels:
            continue
        image_list_temp[ke] = image_list[ke]
    image_list = image_list_temp

    train_set, test_set, cnt, tl = [], [], 0, len(image_list)
    for ke, va in image_list.items():
        cnt += 1
        if cnt <= int(tl * 0.8):
            train_set.append([va, ke, labels[ke]])
        else:
            test_set.append([va, ke, labels[ke]])
    
    print(f'train_set is {len(train_set)}')
    print(f'test_set is {len(test_set)}')
    
    return train_set, test_set

def collate_RL(data):
    img = [i[0].tolist() for i in data]
    name = [i[1] for i in data]
    labels = []
    for i in data:
        one_batch = []
        for j in i[-1]:
            one_batch.append(j)
        labels.append(one_batch)
    return torch.tensor(img), name, torch.tensor(labels)
