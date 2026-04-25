import time
import curses
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(acc):
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    data = {}

    for tag in tags:
        events = acc.Scalars(tag)
        if events:
            last = events[-1]
            data[tag] = (last.step, last.value)
    return data


def draw(stdscr, logdir, refresh):
    curses.curs_set(0)
    stdscr.nodelay(True)

    acc = EventAccumulator(logdir, size_guidance={"scalars": 0})
    acc.Reload()

    while True:
        stdscr.erase()
        stdscr.addstr(0, 0, f"TensorBoard Live Viewer — {logdir}")
        stdscr.addstr(1, 0, "Press q to quit")

        data = load_scalars(acc)

        row = 3
        stdscr.addstr(row, 0, f"{'TAG':40} {'STEP':>10} {'VALUE':>12}")
        stdscr.addstr(row + 1, 0, "-" * 65)
        row += 2

        for tag, (step, value) in sorted(data.items()):
            stdscr.addstr(row, 0, f"{tag:40} {step:10d} {value:12.6f}")
            row += 1

        stdscr.refresh()
        time.sleep(refresh)

        try:
            if stdscr.getkey() == "q":
                break
        except:
            pass


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str)
    parser.add_argument("--refresh", type=float, default=1.0)
    args = parser.parse_args()

    curses.wrapper(draw, args.logdir, args.refresh)


if __name__ == "__main__":
    main()
