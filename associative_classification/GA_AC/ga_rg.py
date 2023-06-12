from typing import Hashable
from ruleitem import RuleItem
from manager import DatasetManager
from random import choice, randint, uniform, choices


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


def init_population(manager: DatasetManager, min_support: float, class_label: Hashable):
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

    # population = [r for r in rules if r.support >= min_support]
    population = sorted(rules)
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

    population = choices(initial_population, k=pop_size)

    auxilary_population = set()

    for _ in range(max_iteration):
        if len(population) == 0:
            return population

        population = sorted(population, key=fit, reverse=True)
        mating_candidates = set(population).union(auxilary_population)
        # roulette_wheel, thresh = create_roulette_wheel(population)

        new_pop = set()
        # offsprings = []

        while len(new_pop) < pop_size:
            if len(population) == 0:
                break

            parent1 = tournament_selection(
                list(mating_candidates), len(mating_candidates) // 5
            )
            parent2 = tournament_selection(
                list(mating_candidates), len(mating_candidates) // 5
            )

            while parent1 == parent2:
                parent2 = tournament_selection(population, len(population) // 5)

            child1, child2 = crossover(parent1, parent2, manager)
            if uniform(0, 1) <= mutation_rate:
                child1 = mutation(child1, manager)

            if uniform(0, 1) <= mutation_rate:
                child2 = mutation(child2, manager)

            new_pop.add(child1)
            new_pop.add(child2)

        if len(new_pop) == 0:
            return population

        auxilary_population = auxilary_population.union(
            {rule for rule in new_pop if rule.support >= min_support }
        )

        print("Gen-fit", sum(x.fitness for x in population) / len(population))
        population = list(new_pop)

    # population = sorted(population, key=fit, reverse=True)
    population = sorted(auxilary_population, key=fit, reverse=True)
    return population


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
        initial_population = init_population(manager, min_support, label)
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
        max_iter=70,
        pop_size=20,
        mutation_rate=0.5,
        min_support=0.05,
        min_confidence=0.5,
    )

    rbc = sorted(rules, reverse=True)
    # cars = [rule for rule in rbc if rule.support >= 0.05 and rule.confidence >= 0.5]
    cars = rbc
    print("Rule number:", len(cars))
    print("Conf = 1:", sum(rule.confidence == 1 for rule in cars))
    for rule in cars:
        print(rule)
