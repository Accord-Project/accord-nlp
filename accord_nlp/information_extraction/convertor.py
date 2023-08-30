# Created By Pouyan Parsafard - 8/24/2023
# Modified by Hansi on 28/08/2023

# import nltk
# nltk.download('averaged_perceptron_tagger')

import re
from collections import Counter
from itertools import combinations

import graphviz
import pandas as pd
from nltk import pos_tag
from nltk.chunk import conlltags2tree
from nltk.tree import Tree


def entity_pairing(sentence, entities):
    """
    Pair entities

    :param sentence: str
    :param entities: list of {entity text: tag}
        e.g. [{'Perimeter': 'B-object'}, {'insulation': 'I-object'}, {'should': 'O'}, {'be': 'O'}, {'continuous': 'B-quality'}]
    :return: dataframe with the following columns
        sentence - input sentence,
        format1 - entities,
        output - sentence with tagged entity pair (e.g. <e1>Perimeter insulation</e1> should be <e2>continuous</e2> .),
        <e1> - text of entity 1,
        <e2> - text of entity 2,
        <e1>_tag - tag of entity 1,
        <e2>_tag - tag of entity 2,
        <e1>_startIndex, <e1>_endIndex, <e2>_startIndex and <e2>_endIndex
    """
    # creating output dataframe
    df_output = pd.DataFrame(columns=['sentence', 'format1', 'output', '<e1>', '<e2>', '<e1>_tag', '<e2>_tag',
                                      '<e1>_startIndex', '<e1>_endIndex', '<e2>_startIndex', '<e2>_endIndex',
                                      'prediction'])

    # finding Tokens and Tags
    tokens = [list(dict(x).keys())[0] for x in entities]
    tags = [list(dict(x).values())[0] for x in entities]

    pos_tags = [pos for token, pos in pos_tag(tokens)]

    # making tree to extract the tags and tokens simultaneously
    conlltags = [(token, pos, tg) for token, pos, tg in zip(tokens, pos_tags, tags)]
    ne_tree = conlltags2tree(conlltags)

    # getting out entities and number of their occurrences and getting ready to find out different combinations
    original_text = []
    for subtree in ne_tree:
        # skipping 'O' tags
        if type(subtree) == Tree:
            original_label = subtree.label()
            original_string = " ".join([token for token, pos in subtree.leaves()])
            original_text.append((original_string, original_label))
            none_entityType_text = [x[0] for x in original_text]
            c = Counter()
            c.update(Counter(none_entityType_text))
            counting_noneEntityType_text = []
            for k, v in c.most_common():
                temp_list = [x for x in original_text if x[0] == k]
                if v == 1:
                    for x in original_text:
                        if x[0] == k:
                            counting_noneEntityType_text.append((k, 0, x[1]))
                else:
                    for j in range(1, v + 1):
                        z = []
                        z.append(k)
                        z.append(j)
                        z.append(temp_list[j - 1][1])
                        z = tuple(z)
                        counting_noneEntityType_text.append(z)

                        # making pair combinations of entities
    combinations_list = list(combinations(counting_noneEntityType_text, 2))

    # making index of entity dictionary --> entity_index_dictionary
    entity_index_dictionary = {}
    for _ in set(none_entityType_text):
        result = re.finditer(_, sentence)
        entity_index_dictionary[_] = {}
        if len([x for x in none_entityType_text if x == _]) > 1:
            counter_below = 1
        else:
            counter_below = 0
        for xx in result:
            entity_index_dictionary[_][counter_below] = xx.span()
            counter_below += 1

    # Final combination list (adding </e> tags + adding index) --> combinations_list_withindex
    combinations_list_withindex = []
    for x in combinations_list:
        # first entity with detail (first tuple)
        y0 = list(x[0])
        y0.append(entity_index_dictionary[x[0][0]][x[0][1]])
        y0 = tuple(y0)

        # second entity with detail (second tuple)
        y1 = list(x[1])
        y1.append(entity_index_dictionary[x[1][0]][x[1][1]])
        y1 = tuple(y1)

        # final output containing </e> tags (third tuple)
        part1 = sentence[: y0[3][0]]
        part2 = sentence[y0[3][1]: y1[3][0]]
        part3 = sentence[y1[3][1]:]
        y2 = part1 + '<e1>' + y0[0] + '</e1>' + part2 + '<e2>' + y1[0] + '</e2>' + part3

        combinations_list_withindex.append((y0, y1, y2))

    df_output_rowNumber = 0
    # Finalizing output dataframe
    for x in combinations_list_withindex:
        df_output.loc[df_output_rowNumber, 'sentence'] = sentence
        df_output.loc[df_output_rowNumber, 'format1'] = entities
        df_output.loc[df_output_rowNumber, 'output'] = x[2]
        df_output.loc[df_output_rowNumber, '<e1>'] = x[0][0]
        df_output.loc[df_output_rowNumber, '<e2>'] = x[1][0]
        df_output.loc[df_output_rowNumber, '<e1>_tag'] = x[0][2]
        df_output.loc[df_output_rowNumber, '<e2>_tag'] = x[1][2]
        df_output.loc[df_output_rowNumber, '<e1>_startIndex'] = x[0][3][0]
        df_output.loc[df_output_rowNumber, '<e1>_endIndex'] = x[0][3][1]
        df_output.loc[df_output_rowNumber, '<e2>_startIndex'] = x[1][3][0]
        df_output.loc[df_output_rowNumber, '<e2>_endIndex'] = x[1][3][1]
        df_output_rowNumber += 1

    return df_output


def graph_building(df, view=True):
    """
    Build graphs based on entity pairs and their relations

    :param df: dataframe with the following columns
        sentence - input sentence,
        format1 - entities,
        output - sentence with tagged entity pair (e.g. <e1>Perimeter insulation</e1> should be <e2>continuous</e2> .),
        <e1> - text of entity 1,
        <e2> - text of entity 2,
        <e1>_tag - tag of entity 1,
        <e2>_tag - tag of entity 2,
        <e1>_startIndex, <e1>_endIndex, <e2>_startIndex, <e2>_endIndex,
        prediction - relation between e1 and e2
    :return: graphviz graph
    """
    # predifined variables - colors for nodes
    dict_entityColors = {'object': '#dae8fc', 'quality': '#e1d5e7', 'property': '#fff2cc', 'value': '#ffffff'}
    sentenceBySentence = True

    # Creating Graph
    graph = graphviz.Graph(format='png')

    # df cleaning
    # taking out none relation
    new_df = df[df['prediction'] != 'none'].reset_index(drop=True)
    # taking out nulls
    new_df = new_df[new_df['prediction'].isnull() == False].reset_index(drop=True)

    # looping over df to draw graph
    if sentenceBySentence:
        for i in range(len(new_df)):
            nodeA = str(new_df.loc[i, '<e1>']) + '\n|<' + str(new_df.loc[i, '<e1>_tag']) + '>' + '\n|' + str(
                new_df.loc[i, '<e1>_startIndex']) + '\n|' + str(new_df.loc[i, 'sentence'])
            nodeB = str(new_df.loc[i, '<e2>']) + '\n|<' + str(new_df.loc[i, '<e2>_tag']) + '>' + '\n|' + str(
                new_df.loc[i, '<e2>_startIndex']) + '\n|' + str(new_df.loc[i, 'sentence'])
            splitted_nodeA = nodeA.split('|')
            splitted_nodeB = nodeB.split('|')
            graph.node(nodeA, label=splitted_nodeA[0] + splitted_nodeA[1], style='filled',
                       fillcolor=dict_entityColors[new_df.loc[i, '<e1>_tag']])
            graph.node(nodeB, label=splitted_nodeB[0] + splitted_nodeB[1], style='filled',
                       fillcolor=dict_entityColors[new_df.loc[i, '<e2>_tag']])
            graph.edge(nodeA, nodeB, label=str(new_df.loc[i, 'prediction']))

    if view:
        graph.render(view=view)

    # st.graphviz_chart(graph)
    return graph
