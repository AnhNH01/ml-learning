from typing import Hashable
from ruleitem import RuleItem
from manager import DatasetManager
from random import choice, randint, uniform, choices
import logging


logger = logging.getLogger('ga-rg')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("/home/haianh/grad_project/ml-learning/associative_classification/GA_AC/ga_rg.log")
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


class CARs:
    """
    Hold class association rules that satisfy minimum support

    Methods:
        append: merge two sets of class assciation rules

        sorted_by_fitness: Sort by rules by their fitness, descending order
    """

    def __init__(self):
        self.rules: set[RuleItem] = set()

    def _add(self, rule: RuleItem, min_confidence: float):
        if rule.confidence >= min_confidence:
            self.rules.add(rule)

    def append(self, cars):
        self.rules = self.rules.union(cars.rules)

    def sorted_by_fitness(self):
        def comp_key(rule: RuleItem):
            return rule.fitness

        return sorted(self.rules, key=comp_key, reverse=True)


def mutation(rule: RuleItem, manager: DatasetManager):
    attr = choice(list(manager.items.keys()))
    val = choice(list(manager.items[attr]))

    rule.cond_set[attr] = val
    return RuleItem(rule.cond_set, rule.class_label, manager)


def crossover(parent_one: RuleItem, parent_two: RuleItem, manager: DatasetManager):
    crossover_point = randint(0, parent_one.cond_set_length - 1)
    cond_set1 = (
        parent_one.cond_set[:crossover_point] + parent_two.cond_set[crossover_point:]
    )
    cond_set2 = (
        parent_two.cond_set[:crossover_point] + parent_one.cond_set[crossover_point:]
    )
    child_one = RuleItem(cond_set1, parent_one.class_label, manager)
    child_two = RuleItem(cond_set2, parent_two.class_label, manager)
    return child_one, child_two


def init_population(manager: DatasetManager, min_support: float, class_label: Hashable, pop_size: int):
    """
    Generate initial population for Genetic algorithm

    @params
        manager: Dataset manager, used for creating RuleItems
        pop_size: Size of initial population
    """
    
    def to_cond_set(attr, val):
        # placeholder for cond_set
        cond_set = [None] * (manager.datacase_length - 1)
        cond_set[attr] = val
        return cond_set

    rules: list[RuleItem] = []

    for attr in manager.items.keys():
        for value in manager.items[attr]:
            rule = RuleItem(to_cond_set(attr, value), class_label, manager)
            rules.append(rule)

    population = [r for r in rules if r.support >= min_support]
    population = choices(rules, k=pop_size)
    return population


def genetic_algorithm(
    initial_population,
    max_iteration: int,
    manager: DatasetManager,
    mutation_rate: float,
    pop_size: None,
    min_support: float,
    min_confidence: float
):
    def fit(rule):
        return rule.fitness

    logger.debug(f"inital poplen = {len(initial_population)}")
    population = initial_population

    auxilary_population = set()

    for _ in range(max_iteration):
        if len(population) == 0:
            return population

        # mating_candidates = list(set(population).union(auxilary_population))[:pop_size]
        mating_candidates = population
        roulette_wheel, thresh = create_roulette_wheel(mating_candidates)

        new_pop = set(sorted(mating_candidates, key=fit, reverse=True)[:pop_size//10])
        
        
        timeout = 0 # thresh hold for iter without new valid child (new_pop length doesn't increase)
        while len(new_pop) < pop_size:
            if len(population) == 0:
                break
            
            if timeout > 500:
                break

            prev_length = len(new_pop)
            
            parent1 = roulette_selection(list(mating_candidates), roulette_wheel, thresh)
            parent2 = roulette_selection(list(mating_candidates), roulette_wheel, thresh)

            while parent1 == parent2:
                parent2 = roulette_selection(list(mating_candidates), roulette_wheel, thresh)

            tries = 0
            skip = True

            if uniform(0, 1) > 0.8:
                child1 = parent1
                child2 = parent2
            else:
                child1, child2 = crossover(parent1, parent2, manager)
                while tries < 4 and skip:
                    if child1.cond_set_length == 0 or child2.cond_set_length == 0:
                        child1, child2 = crossover(parent1, parent2, manager)
                    else:
                        skip = False
                        
                    tries += 1

            if skip:
                timeout += 1
                continue

            if uniform(0, 1) <= mutation_rate:
                child1 = mutation(child1, manager)

            if uniform(0, 1) <= mutation_rate:
                child2 = mutation(child2, manager)
                
                
            new_pop.add(child1)
            
            new_pop.add(child2)

            if len(new_pop) == prev_length:
                timeout += 1

            if len(new_pop) == 0:
                break
        
        population = set(population).union(new_pop)
        # population = new_pop
        auxilary_population = auxilary_population.union(
            {rule for rule in population if rule.support >= min_support and rule.confidence >= min_confidence }
        )
        
        # class_coverage = manager.get_dataset_coverage(auxilary_population, initial_population[0].class_label)
        # logger.info(f"Auxi coverage = {class_coverage}")
        # if class_coverage >= 0.95:
        #     break

        # logger.debug(f'Mean fitness: {sum(x.fitness for x in population) / len(population)} at gen {_}')

        population = sorted(population, key=fit, reverse=True)[:pop_size]

    return auxilary_population

def tournament_selection(population: list, tournament_size: int):
    if tournament_size < 2:
        print("Invalid tournament size!")
        return
    candidates = choices(population, k=tournament_size)

    index = candidates.index(max(candidates, key=lambda c: c.fitness))
    return candidates[index]


def roulette_selection(
    population: list[RuleItem], roulette_wheel: list, roll_thresh: float
):
    """
    Selection strategy based on roulette wheel, more fit candidate tends to be selected more
    """
    if len(roulette_wheel) == 0:
        return None
    else:
        roll = uniform(0, roll_thresh)
        for index, value in enumerate(roulette_wheel):
            if roll <= value:
                return population[index]


def create_roulette_wheel(population: list[RuleItem]):
    wheel = []
    for index, rule in enumerate(population):
        if index > 0:
            wheel.append(wheel[index - 1] + rule.fitness)
        else:
            wheel.append(rule.fitness)
    return wheel, wheel[-1]


def ga_genrules(
    manager: DatasetManager,
    max_iter: int,
    pop_size: int,
    mutation_rate: float,
    min_support: float,
    min_confidence: float,
):
    # cars = CARs()
    rule_list = []
    for label in manager.class_labels:
        print("Class:", label)
        initial_population = init_population(manager, min_support, label, pop_size)
        rules = genetic_algorithm(
            initial_population,
            max_iter,
            manager,
            mutation_rate,
            pop_size,
            min_support,
            min_confidence
        )
        rule_list.extend(rules)
    rule_list = sorted(rule_list, key=lambda x: x.fitness, reverse=True)
    return rule_list


if __name__ == "__main__":
    from read import read

    nursery_name = (
        "/home/haianh/grad_project/ml-learning/benchmark-dataset/nursery.names"
    )
    nursery_data = (
        "/home/haianh/grad_project/ml-learning/benchmark-dataset/nursery-train.data"
    )

    # adult_name = "/home/haianh/grad_project/ml-learning/data/adult.names"
    # adult_data = "/home/haianh/grad_project/ml-learning/data/adult.data"

    data, attr, value_type = read(nursery_data, nursery_name)

    manager = DatasetManager(data)

    rules = ga_genrules(
        manager,
        max_iter=50,
        pop_size=50,
        mutation_rate=0.5,
        min_support=0.05,
        min_confidence=0.5,
    )

    rbc = sorted(rules, reverse=True)
    cars = [rule for rule in rbc if rule.support >= 0.05 and rule.confidence >= 0.5]
    # cars = rbc
    print("Rule number:", len(cars))
    print("Conf = 1:", sum(rule.confidence == 1 for rule in cars))
    for rule in cars:
        print(rule)
