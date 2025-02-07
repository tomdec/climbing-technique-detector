
def get_filename(path: str):
    last_directory_seperator = path.rfind('/')
    if last_directory_seperator != -1:
        wo_path = path[last_directory_seperator+1:]
    else:
        wo_path = path

    start_extension = wo_path.rfind('.')
    if start_extension != -1:
        wo_extension = wo_path[:start_extension]
    else:
        wo_extension = wo_path
    
    return wo_extension