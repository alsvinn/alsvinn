
class Direct(object):
    def __init__(self, alsvinn_files, reference_solutions, datasetnames):
        """
        :param alsvinn_files: files generated from alsvinn
        :param reference_solutions: files generated from
        """

        self.alsvinn_files = alsvinn_files
        self.reference_solutions = reference_solutions
        self.datasetnames = datasetnames

    def compare(self, compare_function):
        results = {}
        for name in self.datasetnames:
            results[name] = []

        for k in range(len(self.alsvinn_files)):
            for name in self.datasetnames:
                alsvinn_solution = self.alsvinn_files[k].read_dataset(name)

                reference_solution = self.reference_solutions[k].read_dataset(name)

                results[name].append(compare_function(alsvinn_solution, reference_solution))

        return results
