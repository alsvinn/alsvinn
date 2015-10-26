import copy
def make_configurations(configurations_lists):
    configurations = []
    keys = configurations_lists.keys()
    configuration = {}
    recurse_configurations(configurations, configuration, configurations_lists, keys, 0)
    return configurations
def recurse_configurations(configurations, configuration, configuration_lists, keys, index):
    key = keys[index]
    options = configuration_lists[key]

    if type(options) is str:
        options = [options]

    for option in options:
        configuration[key] = option

        if index < len(keys) - 1:
            recurse_configurations(configurations, configuration, configuration_lists, keys, index + 1)
        else:
            configurations.append(copy.copy(configuration))


