def get_setup_version_and_pattern(setup_content):
    depend_lst, version_lst = [], []
    for l in setup_content:
        if "==" in l:
            lst = (
                l.split("[")[-1]
                .split("]")[0]
                .replace(" ", "")
                .replace('"', "")
                .replace("'", "")
                .split(",")
            )
            for dep in lst:
                if dep != "\n":
                    version_lst.append(dep.split("==")[1])
                    depend_lst.append(dep.split("==")[0])

    version_high_dict = {d: v for d, v in zip(depend_lst, version_lst)}
    return version_high_dict


def get_env_version(env_content):
    read_flag = False
    depend_lst, version_lst = [], []
    for l in env_content:
        if "dependencies:" in l:
            read_flag = True
        elif read_flag:
            lst = l.replace("-", "").replace(" ", "").replace("\n", "").split("=")
            if len(lst) == 2:
                depend_lst.append(lst[0])
                version_lst.append(lst[1])
    return {d: v for d, v in zip(depend_lst, version_lst)}


def update_dependencies(setup_content, version_low_dict, version_high_dict):
    version_combo_dict = {}
    for dep, ver in version_high_dict.items():
        if dep in version_low_dict.keys() and version_low_dict[dep] != ver:
            version_combo_dict[dep] = dep + ">=" + version_low_dict[dep] + ",<=" + ver
        else:
            version_combo_dict[dep] = dep + "==" + ver

    setup_content_new = ""
    pattern_dict = {d: d + "==" + v for d, v in version_high_dict.items()}
    for l in setup_content:
        for k, v in pattern_dict.items():
            if v in l:
                l = l.replace(v, version_combo_dict[k])
        setup_content_new += l
    return setup_content_new


if __name__ == "__main__":
    with open("pyproject.toml", "r") as f:
        setup_content = f.readlines()

    with open("environment.yml", "r") as f:
        env_content = f.readlines()

    setup_content_new = update_dependencies(
        setup_content=setup_content[2:],
        version_low_dict=get_env_version(env_content=env_content),
        version_high_dict=get_setup_version_and_pattern(
            setup_content=setup_content[2:]
        ),
    )

    with open("pyproject.toml", "w") as f:
        f.writelines("".join(setup_content[:2]) + setup_content_new)
