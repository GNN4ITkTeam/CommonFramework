from time import perf_counter

import torch


class DeviceTimer:
    def __init__(self, device):
        self.device = torch.device(device)
        self._start = None
        self._end = None
        self._start_time = None
        self.elapsed = None

    def start(self):
        self.elapsed = None
        if self.device.type == "cuda":
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._start_time = perf_counter()
        return self

    def record_start(self):
        return self.start()

    def stop(self):
        if self.device.type == "cuda":
            self._end.record()
            torch.cuda.synchronize(self.device)
            self.elapsed = self._start.elapsed_time(self._end) / 1000
        else:
            self.elapsed = perf_counter() - self._start_time
        return self.elapsed

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False


def format_timing_table(title, rows):
    metric_width = max(len("Metric"), *(len(metric) for metric, _ in rows))
    avg_width = max(len("Average (s)"), *(len(f"{value:.3f}") for _, value in rows))
    separator = f"+-{'-' * metric_width}-+-{'-' * avg_width}-+"

    lines = [title, separator]
    lines.append(
        f"| {'Metric'.ljust(metric_width)} | {'Average (s)'.rjust(avg_width)} |"
    )
    lines.append(separator)

    for metric, value in rows:
        lines.append(
            f"| {metric.ljust(metric_width)} | {f'{value:.3f}'.rjust(avg_width)} |"
        )

    lines.append(separator)
    return "\n".join(lines)


def format_grouped_timing_table(title, rows):
    section_width = max(len("Section"), *(len(section) for section, _, _, _, _ in rows))
    metric_width = max(len("Metric"), *(len(metric) for _, metric, _, _, _ in rows))
    parent_width = max(len("Parent"), *(len(parent) for _, _, _, _, parent in rows))
    avg_width = max(len("Average (s)"), *(len(f"{value:.3f}") for _, _, value, _, _ in rows))
    pct_width = len("Percent of parent")
    separator = (
        f"+-{'-' * section_width}-+-{'-' * metric_width}-+-{'-' * parent_width}-+-{'-' * avg_width}-+-{'-' * pct_width}-+"
    )

    lines = [title, separator]
    lines.append(
        f"| {'Section'.ljust(section_width)} | {'Metric'.ljust(metric_width)} | {'Parent'.ljust(parent_width)} | {'Average (s)'.rjust(avg_width)} | {'Percent of parent'.rjust(pct_width)} |"
    )
    lines.append(separator)

    current_section = None
    for section, metric, value, parent_value, parent_label in rows:
        if current_section is not None and section != current_section:
            lines.append(separator)

        if parent_value in (None, 0):
            percent_of_parent = "-"
        else:
            percent_of_parent = f"{(100.0 * value / parent_value):.1f}%"
        lines.append(
            f"| {section.ljust(section_width)} | {metric.ljust(metric_width)} | {parent_label.ljust(parent_width)} | {f'{value:.3f}'.rjust(avg_width)} | {percent_of_parent.rjust(pct_width)} |"
        )
        current_section = section

    lines.append(separator)
    return "\n".join(lines)