def map(p_list):
    result=[]
    for i in p_list:
        result.append(numbers_to_strings(i))
    return result


def numbers_to_strings(argument):
    switcher = {
        1: "Business",
        2: "Technology",
        3: "Entertainment",
        4: "Medical",
        5: "Politics",
        6: "Sports",

    }
    return switcher.get(argument, "nothing")