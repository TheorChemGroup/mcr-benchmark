import os, multiprocessing, inspect


def modes_from_flags(current_dir, assembler_dir="", build_flags=""):
    modes = {
        'gnu': { # gnu_env.yml
            'envload': [
                'source /s/ls4/users/knvvv/condainit',
                'conda activate full',
            ],
            'checkpython': "/s/ls4/users/knvvv/bin/ls_conda/envs/full/bin/python",
            'build': [
                f'cd {assembler_dir}',
                function_call_line('build_ringo', './release_assemble.py', *('current', ['current'], build_flags, os.path.join(assembler_dir, 'release_assemble.py'))),
                f'cd {current_dir}',
            ],
        },
        'python_R': { # rpython_env.yml
            'envload': [
                'source /s/ls4/users/knvvv/condainit',
                'conda activate rpyenv'
            ],
            'checkpython': "/s/ls4/users/knvvv/bin/ls_conda/envs/rpyenv/bin/python",
        },

        # For benchmark we used intelpython for better performance
        # It's possible to recreate this env by hand to have all the same packages as 'gnu'
        # (however, since this env cannot be reproduced from YAML, it's easier to use basic python instead)
        # 'intelrdkit': {
        #     'envload': [
        #         'source /s/ls4/users/knvvv/intelinit',
        #         'conda activate rdkit'
        #     ],
        #     'checkpython': "/s/ls4/users/knvvv/.conda/envs/intelpy/bin/python"
        # },
    }

    # Pass basic python as if it is intelpython, because it's easier to setup from YAML files
    if 'intel' not in modes:
        modes['intel'] = modes['gnu']
    return modes

def function_call_line(function_name, script_path, *args):
    script_name = os.path.basename(script_path).replace('.py', '')
    return "python -c 'from {} import {}; {}(*{})'".format(script_name,
                                                           function_name,
                                                           function_name,
                                                           repr(args).replace("'", '\\"'))


def run_separately(script_parts, jobname='Untitled job'):
    # Assume that all elements of script_parts are lists of str
    script_parts = [element for sublist in script_parts for element in sublist]
    script = '&&'.join(script_parts)

    exec_parts = ['bash', '-c', f'"{script}"']
    print(' '.join(exec_parts))
    exit_code = os.system(' '.join(exec_parts))
    assert exit_code == 0, f'nonzero exit code at task "{jobname}"'

def exec(script_path, func=None, funcname=None, env='gnu', args={}):
    # __file__, func=main, env='gnu'
    script_dir = os.path.dirname(os.path.abspath(script_path))
    execution_mode = modes_from_flags(script_dir)[env]

    function_name = None
    if funcname is not None:
        function_name = funcname
    else:
        function_name = func.__name__

    script_parts = [
        execution_mode['envload'],
        [function_call_line(function_name, script_path, *args)]
    ]
    run_separately(script_parts, jobname=f'Execute {function_name}')

def build_ringo(env_name, build_flags, assemble_path):
    # Primary checks
    assert isinstance(build_flags, str)
    assemble_dir = os.path.dirname(os.path.abspath(assemble_path))
    
    # Build ringo
    execution_mode = modes_from_flags(assemble_dir, assembler_dir=assemble_path, build_flags=build_flags)[env_name]
    script_parts = [
        execution_mode['envload'],
        execution_mode['build']
    ]
    run_separately(script_parts, jobname='Ringo build')


class PathHandler:
    def __init__(self, data={}):
        self._paths = data

    def __getitem__(self, key):
        return self._paths[key]
    
    def __setitem__(self, key, value):
        self._paths[key] = value

    def set_mainwd(self, mainwd):
        self._paths = {
            key: os.path.join(mainwd, value) if mainwd not in value else value
            for key, value in self._paths.items()
            
        }

    def load_global(self):
        # Dirs must have prefix ./
        final_paths = {key: value for key, value in self._paths.items()}
        for key, value in final_paths.items():
            if not value.startswith('./'):
                final_paths[key] = os.path.join('.', value)
        
        caller_frame = inspect.currentframe().f_back
        caller_globals = caller_frame.f_globals
        for key, value in final_paths.items():
            if caller_globals[key] is not None:
                continue
            caller_globals[key] = value

def process_chunk(input_data):
    function, xyz_chuck, args = input_data
    for xyz_file in xyz_chuck:
        function(xyz_file, *args)

def parallelize_call(input_data, function, nthreads=4, args=()):
    input_data_chunks = [[] for i in range(nthreads)]
    next_chunk = 0
    for item in input_data:
        input_data_chunks[next_chunk].append(item)
        next_chunk += 1
        if next_chunk == nthreads:
            next_chunk = 0
            
    with multiprocessing.Pool(processes=nthreads) as pool:
        pool.map(process_chunk, [(function, chunk, args) for chunk in input_data_chunks])