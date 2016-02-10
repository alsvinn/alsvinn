import alsvinn.runner
import alsvinn.data
import lp_norm
import numpy
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

    def estimate_convergence_rate(self, resolutions, errors):
        rate = 1e6
        resolutions = [resolution[0] for resolution in resolutions]
        for (key, values) in errors.iteritems():
            rate = min(rate, -numpy.polyfit(numpy.log(resolutions), numpy.log(values),1)[0])
        return rate

def convergence_compare(basefile, reference_solution, configurations, resolutions, required_convergence_rate,
                        datasets_names=alsvinn.EULER_DATASETS):

    reference_reader = alsvinn.data.make_reader(reference_solution)
    convergence_data = {"configurations":[],
                        "convergence" : []}
    for configuration in alsvinn.runner.make_configurations(configurations):
        runner = alsvinn.runner.Alsvinn(basefile, configuration)
        datasets = [alsvinn.data.make_reader(x) for x in runner(resolutions)]



        convergence = Convergence(datasets, reference_reader, datasets_names)

        result = convergence.compare(lp_norm.LpNorm(1))

        rate = convergence.estimate_convergence_rate(resolutions, result)

        if rate < required_convergence_rate:
            raise Exception("Computed convergence rate was %f. Did not reach convergence rate of %f for configuration %s. " % (rate, required_convergence_rate, str(configuration)))

        convergence_data["configurations"].append(configuration)
        convergence_data["convergence"].append(result)


    return convergence_data


