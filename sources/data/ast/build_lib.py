from tree_sitter import Language

import subprocess

subprocess.run(['git', 'clone', f'git@github.com:tree-sitter/tree-sitter-java.git',
                    f'vendor/tree-sitter-java'])

Language.build_library(
    'build/my-languages.so',
    [
        'vendor/tree-sitter-java'
    ]
)