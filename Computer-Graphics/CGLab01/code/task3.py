file = open("../data/model_1.obj")
v = []
for s in file:
    sp = s.split()
    if sp[0] == "v":
        v.append(list(map(float, sp[1:])))
print(v)