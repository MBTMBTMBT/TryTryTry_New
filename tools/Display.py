def print_progress_rate(present_num, total_num: int):
    bar_num = present_num // total_num * 20
    print("\b" * len(" %d / %d" % (present_num - 1, total_num)), end="")
    print("%d / %d" % (present_num, total_num), end="")


def format_time(ms):
    ss = 1000
    mi = ss * 60
    hh = mi * 60

    hours = ms // hh
    minutes = (ms - hours * hh) // mi
    seconds = (ms - hours * hh - minutes * mi) // ss
    milliseconds = ms - hours * hh - minutes * mi - seconds * ss

    return "%dh/%dm/%ds/%dms" % (hours, minutes, seconds, milliseconds)
