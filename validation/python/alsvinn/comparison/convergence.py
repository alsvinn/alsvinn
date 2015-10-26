import alsvinn.runner
import alsvinn.data
import lp_norm

class Convergence(object):
    def __init__(self, alsvinn_files, reference_solution, datasetnames):
        """
        :param alsvinn_files: files generated from alsvinn
        :param reference_solutions: files generated from
        """

        self.alsvinn_files = alsvinn_files
        self.reference_solution = reference_solution
        self.datasetnames = datasetnames

    def compare(self, compare_function):
        results = {}
        for name in self.datasetnames:
            results[name] = []
        for name in self.datasetnames:
            reference_solution = self.reference_solution.read_dataset(name)
            for k in range(len(self.alsvinn_files)):
                alsvinn_solution = self.alsvinn_files[k].read_dataset(name)
                results[name].append(compare_function(alsvinn_solution, reference_solution))

        return results

def convergence_compare(basefile, reference_solution, configurations, resolutions, required_convergence_rate,
                        datasets_names=alsvinn.EULER_DATASETS):

    reference_reader = alsvinn.data.make_reader(reference_solution)
    convergence_data = {"configurations":[],
                        "convergence" : []}
    for configuration in alsvinn.runner.make_configurations(configurations):
        runner = alsvinn.runner.Alsvinn(basefile, configuration)
        output_files = runner(resolutions)

        datasets = [alsvinn.data.make_reader((x, "alsvinn")) for x in output_files]


        convergence = Convergence(datasets, reference_reader, datasets_names)

        result = convergence.compare(lp_norm.LpNorm(1))

        convergence_data["configurations"].append(configuration)
        convergence_data["convergence"].append(result)
    return convergence_data


