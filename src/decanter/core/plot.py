# pylint: disable=too-many-locals
"""Plotting information of experiment

Some function showing chart for easy-compare between
models information in an experiement.

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decanter.core.enums.evaluators import Evaluator
from decanter.core.enums import check_is_enum

def show_model_attr(metric, score_types, exp):
    """Show all models attribute in Experiment

    Args:
        metric(list): List of str indicates metrics ['mse', 'mae'...].
            Informations to show in chart.
        score_types(list): List of str indicates score_type ['cv_averages',
            'validation'...]. Informations to show in chart.
        exp(:class:`~decanter.core.jobs.experiment.Experiemnt`):
            The experiment that want to show its models attributes.
    Returns:
        class pandas.DataFrame
    """
    def get_attr(metric, score_types, exp_attributes):
        """Get all models metric of score_type in exp_attributes"""
        metric = check_is_enum(Evaluator, metric)
        model_names = []
        model_attr_lists = [[] for i in range(len(score_types))]
        for key, val in exp_attributes.items():
            model_names.append(key)
            for i, score_type in enumerate(score_types):
                model_attr_lists[i].append(val[score_type][metric])

        return model_names, model_attr_lists

    type_map = {
        'cv_averages': 'cv_averages',
        'validation': 'validation_scores'
    }

    score_types = [type_map[score_type] for score_type in score_types]
    model_names, attr_lists = get_attr(
        metric=metric, score_types=score_types, exp_attributes=exp.attributes)

    model_labels = score_types

    chart_bars = np.arange(len(model_names))
    width = 0.35

    fig, axes = plt.subplots()
    rects = []
    ymin = ymax = None
    bar_n = len(score_types)
    for i, attr_vals in enumerate(attr_lists):
        axes.bar(
            chart_bars + width/bar_n * i,
            attr_vals, width,
            label=model_labels[i])
        ymin = min(attr_vals) if ymin is None else min(ymin, min(attr_vals))
        ymax = max(attr_vals) if ymax is None else max(ymax, max(attr_vals))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axes.set_ylabel('{} Scores'.format(metric))
    axes.set_xticks(chart_bars)
    axes.set_xticklabels(model_names)
    axes_ = plt.gca()
    axes_.set_ylim(bottom=ymin, top=ymax)
    locs, _ = plt.yticks()
    unit = (locs[1] - locs[0])
    axes_.set_ylim(bottom=ymin - unit, top=ymax + unit)
    axes.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            axes.annotate(
                '%s', height,
                xy=(rect.get_x() + rect.get_width() / bar_n, height),
                xytext=(0, 5),  # 3 points vertical offset
                textcoords='offset points',
                ha='center', va='bottom')

    for rect in rects:
        autolabel(rect)

    fig.tight_layout()

    plt.show()

    columns = ['name', 'id'] + score_types
    model_table = []
    for i, model_name in enumerate(model_names):
        row = [model_name, exp.models[i]]
        for j in range(len(score_types)):
            row.append(attr_lists[j][i])
        model_table.append(row)

    model_table_df = pd.DataFrame(model_table, columns=columns)
    return model_table_df
