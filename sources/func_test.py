from data.data_utils import tokenize_source, parse_for_completion, generate_input
from data.ast.ast_parser import generate_single_ast_nl

SOURCE = "public int[] twoSum(int[] nums, int target) {" \
         "PRED " \
         "for (int i = 0; i < n; ++i) {" \
         "for (int j = i + 1; j < n; ++j) {" \
         "if (nums[i] + nums[j] == target) {" \
         "return new int[]{i, j};}}}" \
         "return new int[0];}"

# test tokenize source code into tokens list
def tokenize_source_test():
    tokens = tokenize_source(SOURCE)
    print(tokens)


# test generate ast and nl from source code
def generate_ast_nl_test():
    ast, nl = generate_single_ast_nl(SOURCE)
    print(ast)
    print('-' * 100)
    print(nl)

# to see what dataset item like
def dataset_item_test():
    source_path = './dataset/source.txt'
    target_path = './dataset/target.txt'
    codes, ast, names, target = parse_for_completion(source_path, target_path)
    print('-'*100)
    print(codes)
    print('-'*100)
    print(ast)
    print('-'*100)
    print(names)
    print('-'*100)
    print(target)