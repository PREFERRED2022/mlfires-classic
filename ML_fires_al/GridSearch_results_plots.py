import ast

lines = []
with open('/home/sgirtsou/Documents/GridSearchCV/fold2_full_params') as f:
    for a in f:
        lines.append(ast.literal_eval(a))


print(lines)