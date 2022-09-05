import sys
import inspect

import common.utils as utils
import pyrenderer


def _print_indented_docs(lines, prefix:str, out):
    # remove empty lines at the start
    num_empty_lines = 0
    for i in range(len(lines)):
        if len(lines[i].strip())!=0:
            break
        num_empty_lines += 1
    lines = lines[num_empty_lines:]
    # remove empty lines at the end
    num_empty_lines = 0
    for i in range(len(lines)-1, 0, -1):
        if len(lines[i].strip()) != 0:
            break
        num_empty_lines += 1
    lines = lines[:-num_empty_lines]
    if len(lines)==0: return
    # collect leading whitespace
    leading_spaces = len(lines[0]) - len(lines[0].lstrip())
    # write out
    print(prefix+'""""')
    for line in lines:
        print(prefix+line[leading_spaces:], file=out)
    print(prefix + '""""')


def print_docs(root, prefix='', out=sys.stdout):
    for name, obj in inspect.getmembers(root):
        if name.startswith('__'): continue
        if inspect.isclass(obj):
            # class name + documentation
            print(prefix+"class", name, ':', file=out)
            if obj.__doc__ is not None:
                _print_indented_docs(obj.__doc__.split('\n'), prefix+'    ', out)
            # member
            print_docs(obj, prefix+'    ', out)
            print("", file=out)
        else:
            # method name + documentation
            if obj.__doc__ is None:
                print(prefix, name, "= ??", file=out) # unknown variable
            else:
                # a method or function
                lines = obj.__doc__.split('\n')
                if len(lines[0])==0:
                    print(prefix+"property", name+" = Unknown", file=out)
                elif lines[0].startswith("Members:"):
                    print(prefix+"enum", name, "= auto()", file=out)
                else:
                    print(prefix+"def", lines[0]+":", file=out)
                    _print_indented_docs(lines[1:], prefix+'    ', out)
                    print("", file=out)


if __name__ == '__main__':
    print_docs(pyrenderer)
