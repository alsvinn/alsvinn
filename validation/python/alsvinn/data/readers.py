import alsvinn.data
def make_reader(file_tuple):
    file_type = file_tuple[1]
    if file_type == 'alsvinn':
        return alsvinn.data.VizSchemaReader(file_tuple[0])
    elif file_type == 'alsvid':
        return alsvinn.data.AlsvidReader(file_tuple[0])
    else:
        raise Exception("unknown format %s" % file_type)
