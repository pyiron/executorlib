import tomlkit


if __name__ == "__main__":
    with open("pyproject.toml", "r") as f:
        data = tomlkit.load(f)

    lst = []
    for sub_lst in data["project"]["optional-dependencies"].values():
        for el in sub_lst:
            lst.append(el)

    data["project"]["dependencies"] += list(set(lst))

    with open("pyproject.toml", "w") as f:
        f.writelines(tomlkit.dumps(data))