from gpustat import GPUStatCollection
from time import sleep
from socket import getfqdn
import argparse
from ansi2html import Ansi2HTMLConverter
from email.message import EmailMessage
from smtplib import SMTP
from os import getlogin, environ
from random import sample
import sys
import subprocess

def _getargparser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="""
            Waits for n GPUs to become available.

            The script checks every few minutes which GPUs are available (i.e., have no 
            associated processes). If at least n GPUs are available, it quits and prints 
            a suitable CUDA_VISIBLE_DEVICES for n GPUs to stdout. It can optionally notify
            interested parties by e-mail.

            Use, e.g., like:

            $(require-gpu 3) && ./start-my-training.sh

            if you need 3 GPUs

            """)
    if p.prog == 'wait-for-gpu': 
        p.add_argument('email', nargs='*', help="Notify these parties by e-mail when the GPUs are available")
        p.add_argument('-n', default=1, nargs='?', type=int, help="Number of GPUs required")
    else:
        p.add_argument('n', default=1, nargs='?', type=int, help="Number of GPUs required")
        p.add_argument('-e', '--email', nargs='*', 
                help="Notify these parties by e-mail when the GPUs are available")
    p.add_argument('-c', '--command', nargs=1, 
            help="""Run a command when the GPUs are available. The given command is run via the system shell
            with the matching CUDA_VISIBLE_DEVICES environment. When used together with --email, a second
            e-mail will be sent when the command has finished. """)
    p.add_argument('-i', '--interval', type=float, default=5, help="Number of minutes to wait between checks")
    p.add_argument('-q', '--quiet', action='store_true', default=False, 
            help="Don’t print gpustat output")
    p.add_argument('-f', '--first', action='store_true', default=False,
            help="Print an export for the first n available gpus instead of selecting gpus randomly")
    p.add_argument('-1', '--once', action='store_true', default=False,
            help="Return immediately without waiting. This will return an exit code != 0 if not enough GPUs are available")
    return p

def available_gpus(query):
    return [gpu for gpu in query if len(gpu.processes) == 0]

def wait_for_gpus(n: int, interval: float, once: bool = False):
    first_time = True
    while True:
        query = GPUStatCollection.new_query()
        gpus = available_gpus(query)
        if len(gpus) >= n:
            return query
        elif once:
            return None
        if first_time:
            print(f'Waiting for {n} free GPUs, checking every {interval} minutes ...',
                    querytostring(query), sep='\n', file=sys.stderr)
            first_time = False
        sleep(60 * interval)

def querytostring(query: GPUStatCollection) -> str:
    return '\n'.join(map(repr, query))

def mail_query(query, recipients, suffix='', subject=None):
    msg = EmailMessage()
    msg['Subject'] = subject or f'{len(available_gpus(query))} GPUs on {getfqdn()} available'
    msg['To'] = ', '.join(recipients)
    msg['From'] = f'{getlogin()}@{getfqdn()}'
    if isinstance(query, GPUStatCollection):
        text = querytostring(query)
    else:
        text = str(query)
    if suffix:
        text += '\n\n' + suffix
    msg.set_content(Ansi2HTMLConverter(dark_bg=False).convert(text), subtype='html', cte='8bit')
    with SMTP('localhost') as smtp:
        smtp.send_message(msg)

def success(query, options):
    if not options.quiet:
        print(querytostring(query), file=sys.stderr)
    raw_ids = [gpu.index for gpu in available_gpus(query)]
    ids = raw_ids[:options.n] if options.first else sample(raw_ids, options.n)
    ids_string = ','.join(map(str, ids))
    export = f'export CUDA_VISIBLE_DEVICES={ids_string}'
    print(export, file=sys.stdout)
    if options.email:
        mail_query(query, options.email, suffix=export)
    if options.command:
        env = environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ids_string
        if len(options.command) == 1:
            result = subprocess.run(options.command[0], shell=True, env=env)
        else:
            result = subprocess.run(options.command, env=env)
        if options.email:
            success = 'succeeded' if result.returncode == 0 else f'failed ({result.returncode})'
            mail_query(f'Command: \t{result.args}\nResult: \t{success}\nHost: \t{getfqdn()}\nGPUs: \t{ids_string}',
                    options.email,
                    subject=f'{result.args} {success}')
        return result.returncode


def main():
    options = _getargparser().parse_args()
    try:
        query = wait_for_gpus(options.n, options.interval, options.once)
    except KeyboardInterrupt:
        print('Cancelled.', file=sys.stderr)
        sys.exit(3)
    if query:
        result = success(query, options)
        sys.exit(result or 0)
    else:
        sys.exit(127)
