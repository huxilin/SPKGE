from multiprocessing.pool import ThreadPool as Pool
from typing import Tuple, Any
from dataset import Dataset
from link_prediction.models.model import Model
from prefilters.prefilter import PreFilter
import numpy as np
from config import MAX_PROCESSES
import threading
import copy
from itertools import chain

class TypeBasedPreFilter(PreFilter):
    """
    The TypeBasedPreFilter object is a PreFilter that relies on the entity types
    to extract the most promising samples for an explanation.
    """
    def __init__(self,
                 model: Model,
                 dataset: Dataset):
        """
        TypeBasedPreFilter object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        """
        super().__init__(model, dataset)

        self.max_path_length = 3
        self.entity_id_2_train_samples = {}
        self.entity_id_2_relation_vector = {}
        self.rel_id_vector = {}
        self.rel_id_vector = model.relation_embeddings.clone().detach().cpu()
        self.entity_id_2_relation_vector =model.entity_embeddings.clone().detach().cpu()

        self.threadLock = threading.Lock()
        self.counter = 0
        self.thread_pool = Pool(processes=MAX_PROCESSES)

        for (h, r, t) in dataset.train_samples:

            if not h in self.entity_id_2_train_samples:
                self.entity_id_2_train_samples[h] = [(h, r, t)]
               # self.entity_id_2_relation_vector[h] = np.zeros(2*self.dataset.num_relations)

            if not t in self.entity_id_2_train_samples:
                self.entity_id_2_train_samples[t] = [(h, r, t)]
               # self.entity_id_2_relation_vector[t] = np.zeros(2*self.dataset.num_relations)

            self.entity_id_2_train_samples[h].append((h, r, t))
            self.entity_id_2_train_samples[t].append((h, r, t))
            # self.entity_id_2_relation_vector[h][r] += 1
            # self.entity_id_2_relation_vector[t][r+self.dataset.num_relations] += 1


    def top_promising_samples_for(self,
                                  sample_to_explain:Tuple[Any, Any, Any],
                                  perspective:str,
                                  top_k=50,
                                  verbose=True):

        """
        This method extracts the top k promising samples for interpreting the sample to explain,
        from the perspective of either its head or its tail (that is, either featuring its head or its tail).

        :param sample_to_explain: the sample to explain
        :param perspective: a string conveying the explanation perspective. It can be either "head" or "tail":
                                - if "head", find the most promising samples featuring the head of the sample to explain
                                - if "tail", find the most promising samples featuring the tail of the sample to explain
        :param top_k: the number of top promising samples to extract.
        :param verbose:
        :return: the sorted list of the k most promising samples.
        """
        self.counter = 0

        if verbose:
            print("Type-based extraction of promising facts for" + self.dataset.printable_sample(sample_to_explain))

        head, relation, tail = sample_to_explain

        if perspective == "head":
            samples_featuring_head = self.entity_id_2_train_samples[head]

            worker_processes_inputs = [(len(samples_featuring_head), sample_to_explain, x, perspective, verbose) for x
                                       in samples_featuring_head]
            results = self.thread_pool.map(self._analyze_sample, worker_processes_inputs)

            sample_featuring_head_2_promisingness = {}

            for i in range(len(samples_featuring_head)):
                sample_featuring_head_2_promisingness[samples_featuring_head[i]] = results[i]

            samples_featuring_head_with_promisingness = sorted(sample_featuring_head_2_promisingness.items(),
                                                               reverse=True,
                                                               key=lambda x: x[1])
            # sorted_promising_samples = [x[0] for x in samples_featuring_head_with_promisingness]
            sorted_promising_samples = samples_featuring_head_with_promisingness

        else:
            samples_featuring_tail = self.entity_id_2_train_samples[tail]

            worker_processes_inputs = [(len(samples_featuring_tail), sample_to_explain, x, perspective, verbose) for x
                                       in samples_featuring_tail]
            results = self.thread_pool.map(self._analyze_sample, worker_processes_inputs)

            sample_featuring_tail_2_promisingness = {}

            for i in range(len(samples_featuring_tail)):
                sample_featuring_tail_2_promisingness[samples_featuring_tail[i]] = results[i]

            samples_featuring_tail_with_promisingness = sorted(sample_featuring_tail_2_promisingness.items(),
                                                               reverse=True,
                                                               key=lambda x: x[1])
            sorted_promising_samples = [x[0] for x in samples_featuring_tail_with_promisingness]

        return sorted_promising_samples[:top_k]


    def _analyze_sample(self, input_data):
        all_samples_number, sample_to_explain, sample_to_analyze, perspective, verbose = input_data

        with self.threadLock:
            self.counter+=1
            i = self.counter

        if verbose:
            print("\tAnalyzing sample " + str(i) + " on " + str(all_samples_number) + ": " + self.dataset.printable_sample(sample_to_analyze))

        head_to_explain, relation_to_explain, tail_to_explain = sample_to_explain
        head_to_analyze, relation_to_analyze, tail_to_analyze = sample_to_analyze

        promisingness = -1
        if perspective == "head":
            if head_to_explain == head_to_analyze:
                promisingness = self._cosine_similarity(tail_to_explain, tail_to_analyze)
            else:
                assert(head_to_explain == tail_to_analyze)
                promisingness = self._cosine_similarity(tail_to_explain, head_to_analyze)

        elif perspective == "tail":
            if tail_to_explain == tail_to_analyze:
                promisingness = self._cosine_similarity(head_to_explain, head_to_analyze)
            else:
                assert (tail_to_explain == head_to_analyze)
                promisingness = self._cosine_similarity(head_to_explain, tail_to_analyze)

        return promisingness

    def _cosine_similarity(self, entity1_id, entity2_id):
        entity1_vector = self.entity_id_2_relation_vector[entity1_id]
        entity2_vector = self.entity_id_2_relation_vector[entity2_id]
        return np.inner(entity1_vector, entity2_vector) / (np.linalg.norm(entity1_vector) * np.linalg.norm(entity2_vector))
    def _cosine_similarity_rel(self, rel1_id, rel2_id):
        rel1_vector = self.rel_id_vector[rel1_id]
        rel2_vector = self.rel_id_vector[rel2_id]
        return abs(np.inner(rel1_vector, rel2_vector)) / (np.linalg.norm(rel1_vector) * np.linalg.norm(rel2_vector))
    def _instent_p(self,entity1_id,entity2_id,rel_vector):
        entity1_vector = self.entity_id_2_relation_vector[entity1_id]
        entity2_vector = self.entity_id_2_relation_vector[entity2_id]
        return entity1_vector+rel_vector-entity_vector
    def top_promising_samples_for_sem(self,
                                  sample_to_explain:Tuple[Any, Any, Any],
                                  perspective:str,
                                  top_k=50,
                                  verbose=True):

        self.counter = 0

        head, relation, tail = sample_to_explain
        print('要解释的三元组ID')
        print(sample_to_explain)

        start_entity, end_entity = (head, tail) if perspective == "head" else (tail, head)

        samples_featuring_start_entity = self.entity_id_2_train_samples[start_entity]

        worker_processes_inputs = [(len(samples_featuring_start_entity),
                                    start_entity, end_entity, samples_featuring_start_entity[i], verbose)
                                   for i in range(len(samples_featuring_start_entity))]
        # 拓扑结构
        results = self.thread_pool.map(self._analyze_sample_sem, worker_processes_inputs)
        results = sorted(results, key=lambda x: x[0])
        results = list(map(lambda t: t[1], results))
        results = list(filter(lambda x: x != ['None'], results))
        print('找出来头尾实体间' + str(self.max_path_length) + '跳内所有路径：' + str(len(results)) + '条')
        # 获取路径内实体
        #all_samples = []
        all_entity = {}
        all_relation = {}
        for i, path in enumerate(results):
            path_entity = []
            path_relation = []
            for j, sample in enumerate(path):
                #all_samples.append(sample)
                path_entity.append(sample[0])
                path_entity.append(sample[2])
                path_relation.append(sample[1])
            all_entity[i] = self.unique_array(path_entity, head)
            all_relation[i] = path_relation
        #all_unique_samples = []
        #for i in all_samples:
        #    if i in all_unique_samples:
        #        continue
        #    else:
        #        all_unique_samples.append(i)
        #entity_cos = []
        #for entity_sample in all_unique_samples:
        #    entity_cos.append((self._cosine_similarity(head, entity_sample[0])+self._cosine_similarity(tail, entity_sample[2]))/2)
        #sample_cos = dict(zip(all_unique_samples, entity_cos))
        #sample_cos_sim = sorted(sample_cos.items(), key=lambda x: x[1], reverse=True)
        #if len(sample_cos_sim)==0:
        #    cos_results_entity = []
        #else:
        #    cos_result_entity = sample_cos_sim[0]
        #    sample_name = self.dataset.sample_to_fact(cos_result_entity[0])
        #    cos_results_entity = [(sample_name,cos_result_entity[1])]
        
        path_relvance =[]
        path_entity_relvance = []
        path_rel_relvance = []
        path_all_relvance = []
        #for rel in all_relation:
            #rels = all_relation[rel]
            #if (len(rels) == 1):
            #    relcos =  self._cosine_similarity_rel(rels[0],relation)
            #else:
            #    relcos = 0
            #if (len(rels) == 2):
            #    relcos =  (self._cosine_similarity_rel(rels[0],relation)+ self._cosine_similarity_rel(rels[1],relation))/2
            #if (len(rels) == 3):
            #    relcos = ( self._cosine_similarity_rel(rels[0],relation) +  self._cosine_similarity_rel(rels[1],relation) +  self._cosine_similarity_rel(rels[2],relation))/3
            #path_rel_relvance.append(relcos)
        for entity in all_entity:
            samples = all_entity[entity]
            if (len(samples) == 2):
            #    cos = 1
                cos = 1
                #cos =  self._cosine_similarity(samples[0],samples[1])
                cos_eve = [cos]
            if(len(samples)==3):
                cos1 = self._cosine_similarity(samples[0], samples[0])
                cos2 = self._cosine_similarity(samples[1], samples[2])
                cos3 = self._cosine_similarity(samples[0], samples[1])
                cos4 = self._cosine_similarity(samples[2], samples[2])
                cos = (cos1 + cos2 + cos3 + cos4)/4
                cos_eve = [cos1,cos2]
            elif(len(samples)==4):
                cos1 = self._cosine_similarity(samples[0], samples[0])
                cos2 = self._cosine_similarity(samples[1], samples[3])
                cos3 = self._cosine_similarity(samples[0], samples[1])
                cos4 = self._cosine_similarity(samples[2], samples[3])
                cos5 = self._cosine_similarity(samples[0], samples[2])
                cos6 = self._cosine_similarity(samples[3], samples[3])
                #cos6 = (cos2 + cos4)/2
                cos = (cos1 + cos2 + cos3 + cos4 + cos5 + cos6) / 6
                cos_eve = [cos1, cos3, cos5]
            path_relvance.append(cos)
            path_all_relvance.append(cos_eve)
        #path_relvance = []
        #for i in range(len(path_rel_relvance)):
        #    sum = path_rel_relvance[i]+path_entity_relvance[i]
        #    path_relvance.append(sum)
        sample_name = []
        for i, path in enumerate(results):
            eve_sample = ()
            for j, sample_id in enumerate(path):
                sample_to_name = self.dataset.sample_to_fact(sample_id)
                eve_sample += sample_to_name
            sample_name.append(eve_sample)

        path_name_relvance = dict(zip(sample_name, path_relvance))
        cos_results = sorted(path_name_relvance.items(), key=lambda x: x[1], reverse=True)
        cos_results_three = cos_results[0:1]
        #three_sample = []
        #three_value = 0
        #for i, sample in enumerate(cos_results_three):
        #    key,value = sample
            #key = key[0:3]
        #    three_sample.append(key)
        #    three_value += value
        #three_sample = tuple(chain.from_iterable(three_sample))
        #cos_results_three = [(three_sample,three_value)]
        #print(cos_results_three)

        return cos_results_three
        #return cos_results_entity

    def unique_array(self,arr,head_id):
        seen = {}
        result = [head_id]
        for item in arr:
            if item != head_id:
                if item not in seen:
                    seen[item] = True
                    result.append(item)
        return result

    def _analyze_sample_sem(self, input_data):
        all_samples_number, start_entity, end_entity, sample_to_analyze, verbose = input_data

        with self.threadLock:
            self.counter += 1
            i = self.counter

        if verbose:
            print("\tAnalyzing sample " + str(i) + " on " + str(
                all_samples_number) + ": " + self.dataset.printable_sample(sample_to_analyze))

        sample_to_analyze_head, sample_to_analyze_relation, sample_to_analyze_tail = sample_to_analyze

        cur_path_length = 1
        next_step_incomplete_paths = []  # each incomplete path is a couple (list of triples in this path, accretion entity)

        # if the sample to analyze is already a path from the start entity to the end entity,
        # then the shortest path length is 1 and you can move directly to the next sample to analyze
        if (sample_to_analyze_head == start_entity and sample_to_analyze_tail == end_entity) or \
                (sample_to_analyze_tail == start_entity and sample_to_analyze_head == end_entity):
            return cur_path_length, \
                [(sample_to_analyze_head, sample_to_analyze_relation, sample_to_analyze_tail)]

        initial_accretion_entity = sample_to_analyze_tail if sample_to_analyze_head == start_entity else sample_to_analyze_head
        next_step_incomplete_paths.append(([sample_to_analyze], initial_accretion_entity))

        # this set contains the entities seen so far in the search.
        # we want to avoid any loops / unnecessary searches, so it is not allowed for a path
        # to visit an entity that has already been featured by another path
        # (that is, another path that has either same or smaller size!)
        entities_seen_so_far = {start_entity, initial_accretion_entity}

        terminate = False
        while not terminate:
            cur_path_length += 1

            cur_step_incomplete_paths = next_step_incomplete_paths
            next_step_incomplete_paths = []

            # print("\t\tIncomplete paths of length " + str(cur_path_length - 1) + " to analyze: " + str(len(cur_step_incomplete_paths)))
            # print("\t\tExpanding them to length: " + str(cur_path_length))
            for (incomplete_path, accretion_entity) in cur_step_incomplete_paths:
                samples_featuring_accretion_entity = self.entity_id_2_train_samples[accretion_entity]

                # print("\tCurrent path: " + str(incomplete_path))

                for (cur_head, cur_rel, cur_tail) in samples_featuring_accretion_entity:

                    cur_incomplete_path = copy.deepcopy(incomplete_path)

                    # print("\t\tCurrent accretion path: " + self.dataset.printable_sample((cur_h, cur_r, cur_t)))
                    if (cur_head == accretion_entity and cur_tail == end_entity) or (
                            cur_tail == accretion_entity and cur_head == end_entity):
                        cur_incomplete_path.append((cur_head, cur_rel, cur_tail))
                        return cur_path_length, cur_incomplete_path

                    # ignore self-loops
                    if cur_head == cur_tail:
                        # print("\t\t\tMeh, it was just a self-loop!")
                        continue

                    # ignore facts that would re-connect to an entity that is already in this path
                    next_step_accretion_entity = cur_tail if cur_head == accretion_entity else cur_head
                    if next_step_accretion_entity in entities_seen_so_far:
                        # print("\t\t\tMeh, it led to a loop in this path!")
                        continue

                    cur_incomplete_path.append((cur_head, cur_rel, cur_tail))
                    next_step_incomplete_paths.append((cur_incomplete_path, next_step_accretion_entity))
                    entities_seen_so_far.add(next_step_accretion_entity)
                    # print("\t\t\tThe search continues")

            if terminate is not True:
                if cur_path_length == self.max_path_length or len(next_step_incomplete_paths) == 0:
                    return 1e6, ["None"]
