def add_binary_argument(parser, arg_name, help=None, default=True):
    parser.add_argument(
        '--{}'.format(arg_name), dest=arg_name,
        help=help,
        action='store_true'
    )
    parser.add_argument(
        '--no-{}'.format(arg_name), dest=arg_name,
        help='do not {}'.format(help) if help is not None else None,
        action='store_false'
    )
    parser.set_defaults(**{arg_name: default})