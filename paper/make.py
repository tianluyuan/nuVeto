#!/usr/bin/env python
import argparse
import paper


if __name__ == '__main__':
    figs = [fn for fn in dir(paper) if fn.startswith('fig_')]
    parser = argparse.ArgumentParser(description='Make plots for paper')
    parser.add_argument('-c', default=None, choices=figs, dest='choice',
                        help='Choice of figure to make. Defaults to all.')
    args = parser.parse_args()

    if args.choice is None:
        for fig in figs:
            getattr(paper, fig)()
    else:
        getattr(paper, args.choice)()
