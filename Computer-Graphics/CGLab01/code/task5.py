file = open("../data/model_1.obj")
f = []
for s in file:
    sp = s.split()
    if sp[0] == "f":
        f_current = []
        for i in sp[1:]:
            f_current.append(int(i.split("/")[0]))
        f.append(f_current)
print(f)