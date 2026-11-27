import tomlkit


if __name__ == "__main__":
    with open("pyproject.toml", "r") as f:
        data = tomlkit.load(f)

    lst = list(data['build-system']['requires'])
    for key, sub_lst in data["project"]["optional-dependencies"].items():
        if key != "docs":
            for el in sub_lst:
                lst.append(el)

    data["project"]["dependencies"] += list(set(lst))

    with open("pyproject.toml", "w") as f:
        f.writelines(tomlkit.dumps(data))
