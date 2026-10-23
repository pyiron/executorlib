import json


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
                dep = dep.strip()
                if dep != "":
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
            lst = l.replace("- ", "").replace(" ", "").replace("\n", "").split("=")
            if len(lst) == 2:
                depend_lst.append(lst[0])
                version_lst.append(lst[1])
    return {d: v for d, v in zip(depend_lst, version_lst)}


def get_next_minor_version(version):
    """Return the next minor release of ``version``, e.g. 2.5.3 -> 2.6.0.

    Falls back to ``None`` when the minor component is not an integer, in which
    case the caller keeps the conservative ``<=pin`` cap.
    """
    parts = version.split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return None
    return parts[0] + "." + str(int(parts[1]) + 1) + ".0"


def update_dependencies(setup_content, version_low_dict, version_high_dict):
    version_combo_dict = {}
    for dep, ver in version_high_dict.items():
        if dep in version_low_dict.keys() and version_low_dict[dep] != ver:
            upper = get_next_minor_version(ver)
            if upper is None:
                upper_cap = ",<=" + ver
            else:
                upper_cap = ",<" + upper
            version_combo_dict[dep] = dep + ">=" + version_low_dict[dep] + upper_cap
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


def convert_key(key, convert_dict):
    if key not in convert_dict.keys():
        return key
    else:
        return convert_dict[key]


if __name__ == "__main__":
    with open(".ci_support/pypi_vs_conda_names.json", "r") as f:
        name_conversion_dict = {v: k for k, v in json.load(f).items()}

    with open("pyproject.toml", "r") as f:
        setup_content = f.readlines()

    with open("environment.yml", "r") as f:
        env_content = f.readlines()

    env_version_dict = {
        convert_key(key=k, convert_dict=name_conversion_dict): v
        for k, v in get_env_version(env_content=env_content).items()
    }

    setup_content_new = update_dependencies(
        setup_content=setup_content[2:],
        version_low_dict=env_version_dict,
        version_high_dict=get_setup_version_and_pattern(
            setup_content=setup_content[2:]
        ),
    )

    with open("pyproject.toml", "w") as f:
        f.writelines("".join(setup_content[:2]) + setup_content_new)
