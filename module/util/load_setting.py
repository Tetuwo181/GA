#coding: UTF-8
import json
import os
import select as SELECT
import crossover as CROSS
import mutation as MUTATION
import birth as BIR
import csv

def load_setting(file_path):
    with open(file_path, "r") as setting_file:
        raw_setting = json.load(setting_file)
    return raw_setting

def load_individual_builder_from_problem(raw_setting, problem):
    gene_length = raw_setting["individual"]["gene_length"]
    gene_base = [0 for index in range(gene_length)]
    answer_base = problem(gene_base)
    return INDI.individual_builder(hasattr(answer_base, "__iter__"))


def load_select(raw_setting):
    setting = raw_setting["select"]
    algo_base = SELECT.get_selection(setting["type"])
    if setting["type"] == "TOUR":
        return algo_base(setting["tournament_num"])
    else:
        return algo_base

def load_crossover(raw_setting):
    genotype = raw_setting["individual"]["genotype"]
    setting = raw_setting["crossover"]
    algo_base = CROSS.get_algo_from_genotype(genotype)
    if algo_base is None:
        return None
    if genotype == "BIN":
        return CROSS.set_prob(algo_base[setting["type"]], setting["prob"] )
    if genotype == "REAL":
        max_list, min_list = load_max_min(raw_setting)
        if setting["type"] == "blx":
            return CROSS.set_prob(algo_base[setting["type"]](min_list, max_list, setting["alpha"]), float(setting["prob"]))
        else:
            return CROSS.set_prob(algo_base[setting["type"]](min_list, max_list), float(setting["prob"]))
    if genotype == "ORD":
        if setting["type"] == "REP":
            return CROSS.set_prob(algo_base[setting["type"]](int(setting["replace_num"])), float(setting["prob"]) )
        else: return CROSS.set_prob(algo_base[setting["type"]], float(setting["prob"]))
    return None

        
def load_max_min(raw_setting):
    if ("max_min" not in raw_setting) or (os.path.exists(raw_setting["max_in"]) is False) :
        min_list = np.array([0 for index in range(int(raw_setting["individual"]["gene_length"]))])
        max_list = np.array([1 for index in range(int(raw_setting["individual"]["gene_length"]))])
    else:
        with open(raw_setting["max_min"], "r") as raw_list:
            data_list = csv.reader(raw_list)
            min_list = np.array([float(data[0]) for data in data_list])
            max_list = np.array([float(data[1]) for data in data_list])
    return (min_list, max_list)


